/*
 * fast_weight_load.c — Zero-copy expert weight loading into MLX Metal buffers
 *
 * Eliminates the np.frombuffer → mx.array Python bottleneck by:
 * 1. Pre-allocating mx.array objects (Metal-backed unified memory)
 * 2. Obtaining raw CPU pointers via Python buffer protocol
 * 3. Using pread() to write directly into Metal buffers from pthreads
 *
 * On Apple Silicon, unified memory means pread() into a Metal buffer's
 * CPU pointer makes data immediately visible to the GPU — no upload step.
 *
 * API:
 *   init(num_workers=8)
 *   prealloc(num_layers, experts_per_layer, components, mx_module)
 *   load_experts(load_list)
 *   shutdown()
 *   stats()
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdatomic.h>

/* macOS QoS */
#include <sys/qos.h>
#include <pthread/qos.h>

/* ---- Constants ---- */

#define FWL_MAX_LAYERS       64
#define FWL_MAX_SLOTS        16      /* expert slots per layer (K parameter) */
#define FWL_MAX_COMPONENTS   9       /* gate/up/down × weight/scales/biases */
#define FWL_MAX_WORKERS      32
#define FWL_MAX_WORK_ITEMS   (FWL_MAX_LAYERS * FWL_MAX_SLOTS * FWL_MAX_COMPONENTS)
#define FWL_COMP_NAME_LEN    48
#define FWL_PAGE_SIZE        16384   /* macOS ARM64 page size */

/* ---- Component spec (one per weight tensor within an expert) ---- */

typedef struct {
    char name[FWL_COMP_NAME_LEN];   /* e.g. "gate_proj.weight" */
    size_t offset;                   /* byte offset within expert block in packed file */
    size_t size;                     /* byte size of this component */
    int ndim;
    int shape[4];
    /* Stored as string for passing to mx: "uint32", "bfloat16", etc. */
    char mx_dtype_str[16];
} ComponentSpec;

/* ---- Pre-allocated buffer entry ---- */

typedef struct {
    PyObject *mx_array;     /* The pre-allocated mx.array (owned reference) */
    void     *data_ptr;     /* Raw CPU pointer into the Metal buffer */
    size_t    byte_size;    /* Total bytes in this buffer */
} PreallocBuf;

/* ---- Per-layer file descriptor ---- */

typedef struct {
    int fd;                 /* fd for layer_XX.bin */
} LayerFd;

/* ---- Single pread work item ---- */

typedef struct {
    int    fd;              /* file descriptor */
    void  *dest;            /* destination pointer (into Metal buffer) */
    size_t nbytes;          /* bytes to read */
    off_t  file_offset;     /* offset in the packed file */
    int    error;           /* errno if failed, 0 on success */
    ssize_t bytes_read;     /* actual bytes read */
} WorkItem;

/* ---- Worker thread context ---- */

typedef struct {
    pthread_t       thread;
    int             worker_id;
    int             running;

    /* Work for current batch */
    WorkItem       *items;
    int             item_count;

    /* Synchronization */
    pthread_mutex_t work_mutex;
    pthread_cond_t  work_cond;
    int             has_work;

    /* Shared completion */
    pthread_mutex_t *done_mutex;
    pthread_cond_t  *done_cond;
    atomic_int      *completed_count;
} WorkerCtx;

/* ---- Module state ---- */

typedef struct {
    /* Worker pool */
    WorkerCtx      *workers;
    int             num_workers;

    /* Pre-allocated buffers: [layer][slot][component] */
    PreallocBuf     bufs[FWL_MAX_LAYERS][FWL_MAX_SLOTS][FWL_MAX_COMPONENTS];

    /* Component specs */
    ComponentSpec   comp_specs[FWL_MAX_COMPONENTS];
    int             num_comps;
    size_t          expert_size;    /* total bytes per expert block */

    /* Layer file descriptors */
    LayerFd         layer_fds[FWL_MAX_LAYERS];
    int             num_layers;
    int             experts_per_layer;

    /* Python references to mx.arrays (organized as nested dicts for Python access) */
    PyObject       *py_buffers;     /* Python dict: buffers[layer][slot][comp_name] = mx.array */

    /* Completion sync */
    pthread_mutex_t done_mutex;
    pthread_cond_t  done_cond;
    atomic_int      completed_count;

    int             initialized;
    int             preallocated;

    /* Stats */
    long long       total_loads;
    long long       total_bytes;
} ModuleState;

static ModuleState g_state = {0};

/* ---- Worker thread function ---- */

static void *worker_func(void *arg) {
    WorkerCtx *ctx = (WorkerCtx *)arg;

    /* Set QoS to USER_INITIATED (latency-sensitive I/O) */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);

    while (ctx->running) {
        pthread_mutex_lock(&ctx->work_mutex);
        while (!ctx->has_work && ctx->running) {
            pthread_cond_wait(&ctx->work_cond, &ctx->work_mutex);
        }
        if (!ctx->running) {
            pthread_mutex_unlock(&ctx->work_mutex);
            break;
        }

        WorkItem *items = ctx->items;
        int count = ctx->item_count;
        ctx->has_work = 0;
        pthread_mutex_unlock(&ctx->work_mutex);

        /* Execute pread calls — no GIL, no Python involvement */
        for (int i = 0; i < count; i++) {
            WorkItem *wi = &items[i];
            ssize_t nread = pread(wi->fd, wi->dest, wi->nbytes, wi->file_offset);
            if (nread < 0) {
                wi->error = errno;
                wi->bytes_read = -1;
            } else if ((size_t)nread < wi->nbytes) {
                /* Short read — try to complete */
                size_t total = (size_t)nread;
                while (total < wi->nbytes) {
                    ssize_t n = pread(wi->fd,
                                      (char *)wi->dest + total,
                                      wi->nbytes - total,
                                      wi->file_offset + (off_t)total);
                    if (n <= 0) {
                        wi->error = (n < 0) ? errno : EIO;
                        wi->bytes_read = (ssize_t)total;
                        break;
                    }
                    total += (size_t)n;
                }
                if (total == wi->nbytes) {
                    wi->error = 0;
                    wi->bytes_read = (ssize_t)total;
                }
            } else {
                wi->error = 0;
                wi->bytes_read = nread;
            }
        }

        /* Signal completion */
        atomic_fetch_add(ctx->completed_count, count);
        pthread_mutex_lock(ctx->done_mutex);
        pthread_cond_signal(ctx->done_cond);
        pthread_mutex_unlock(ctx->done_mutex);
    }

    return NULL;
}

/* ---- init(num_workers=8) ---- */

static PyObject *fwl_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"num_workers", NULL};
    int num_workers = 8;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &num_workers))
        return NULL;

    if (g_state.initialized) {
        PyErr_SetString(PyExc_RuntimeError, "fast_weight_load already initialized");
        return NULL;
    }

    if (num_workers < 1 || num_workers > FWL_MAX_WORKERS) {
        PyErr_Format(PyExc_ValueError, "num_workers must be 1-%d", FWL_MAX_WORKERS);
        return NULL;
    }

    /* Initialize completion sync */
    pthread_mutex_init(&g_state.done_mutex, NULL);
    pthread_cond_init(&g_state.done_cond, NULL);
    atomic_store(&g_state.completed_count, 0);

    /* Create worker threads */
    g_state.num_workers = num_workers;
    g_state.workers = (WorkerCtx *)calloc(num_workers, sizeof(WorkerCtx));
    if (!g_state.workers) {
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < num_workers; i++) {
        WorkerCtx *w = &g_state.workers[i];
        w->worker_id = i;
        w->running = 1;
        w->has_work = 0;
        w->items = NULL;
        w->item_count = 0;
        w->done_mutex = &g_state.done_mutex;
        w->done_cond = &g_state.done_cond;
        w->completed_count = &g_state.completed_count;

        pthread_mutex_init(&w->work_mutex, NULL);
        pthread_cond_init(&w->work_cond, NULL);

        int rc = pthread_create(&w->thread, NULL, worker_func, w);
        if (rc != 0) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to create worker thread %d: %s", i, strerror(rc));
            /* Cleanup threads already created */
            for (int j = 0; j < i; j++) {
                g_state.workers[j].running = 0;
                pthread_mutex_lock(&g_state.workers[j].work_mutex);
                g_state.workers[j].has_work = 1;
                pthread_cond_signal(&g_state.workers[j].work_cond);
                pthread_mutex_unlock(&g_state.workers[j].work_mutex);
                pthread_join(g_state.workers[j].thread, NULL);
                pthread_mutex_destroy(&g_state.workers[j].work_mutex);
                pthread_cond_destroy(&g_state.workers[j].work_cond);
            }
            free(g_state.workers);
            g_state.workers = NULL;
            return NULL;
        }
    }

    g_state.initialized = 1;
    g_state.total_loads = 0;
    g_state.total_bytes = 0;

    Py_RETURN_NONE;
}

/* ---- Helper: get data pointer from an mx.array via buffer protocol ---- */

static void *get_mx_array_ptr(PyObject *mx_array, Py_buffer *view) {
    /*
     * mx.array supports the Python buffer protocol.
     * PyObject_GetBuffer gives us a CPU-accessible pointer into the
     * Metal-backed unified memory buffer. On Apple Silicon this IS
     * the GPU-visible memory — same physical pages.
     */
    if (PyObject_GetBuffer(mx_array, view, PyBUF_WRITABLE | PyBUF_SIMPLE) != 0) {
        /* Try read-only if writable fails */
        PyErr_Clear();
        if (PyObject_GetBuffer(mx_array, view, PyBUF_SIMPLE) != 0) {
            return NULL;
        }
    }
    return view->buf;
}

/*
 * ---- prealloc(num_layers, experts_per_layer, components, packed_dir) ----
 *
 * Pre-allocate mx.array buffers for all expert weight slots.
 *
 * components: list of dicts, each with:
 *   - name: str (e.g. "gate_proj.weight")
 *   - offset: int (byte offset within expert block)
 *   - size: int (byte size)
 *   - shape: list of int
 *   - dtype: str (e.g. "uint32", "bfloat16", "uint16")
 *
 * packed_dir: str path to directory with layer_XX.bin files
 *
 * Returns: nested dict buffers[layer][slot][comp_name] = mx.array
 */

static PyObject *fwl_prealloc(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"num_layers", "experts_per_layer", "components",
                             "packed_dir", "expert_size", NULL};
    int num_layers = 0;
    int experts_per_layer = 0;
    PyObject *py_components = NULL;
    const char *packed_dir = NULL;
    size_t expert_size = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiOs|n", kwlist,
                                     &num_layers, &experts_per_layer,
                                     &py_components, &packed_dir, &expert_size))
        return NULL;

    if (!g_state.initialized) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() first");
        return NULL;
    }

    if (g_state.preallocated) {
        PyErr_SetString(PyExc_RuntimeError, "Buffers already pre-allocated. Call shutdown() first.");
        return NULL;
    }

    if (num_layers <= 0 || num_layers > FWL_MAX_LAYERS) {
        PyErr_Format(PyExc_ValueError, "num_layers must be 1-%d, got %d",
                     FWL_MAX_LAYERS, num_layers);
        return NULL;
    }
    if (experts_per_layer <= 0 || experts_per_layer > FWL_MAX_SLOTS) {
        PyErr_Format(PyExc_ValueError, "experts_per_layer must be 1-%d, got %d",
                     FWL_MAX_SLOTS, experts_per_layer);
        return NULL;
    }

    if (!PyList_Check(py_components)) {
        PyErr_SetString(PyExc_TypeError, "components must be a list");
        return NULL;
    }

    Py_ssize_t num_comps = PyList_Size(py_components);
    if (num_comps <= 0 || num_comps > FWL_MAX_COMPONENTS) {
        PyErr_Format(PyExc_ValueError, "components count must be 1-%d", FWL_MAX_COMPONENTS);
        return NULL;
    }

    /* Parse component specs */
    g_state.num_comps = (int)num_comps;
    for (Py_ssize_t ci = 0; ci < num_comps; ci++) {
        PyObject *comp = PyList_GetItem(py_components, ci);
        if (!comp || !PyDict_Check(comp)) {
            PyErr_Format(PyExc_TypeError, "components[%zd] must be a dict", ci);
            return NULL;
        }

        ComponentSpec *cs = &g_state.comp_specs[ci];

        /* name */
        PyObject *py_name = PyDict_GetItemString(comp, "name");
        if (!py_name) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'name'", ci);
            return NULL;
        }
        const char *name = PyUnicode_AsUTF8(py_name);
        if (!name) return NULL;
        strncpy(cs->name, name, FWL_COMP_NAME_LEN - 1);
        cs->name[FWL_COMP_NAME_LEN - 1] = '\0';

        /* offset */
        PyObject *py_off = PyDict_GetItemString(comp, "offset");
        if (!py_off) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'offset'", ci);
            return NULL;
        }
        cs->offset = (size_t)PyLong_AsUnsignedLongLong(py_off);

        /* size */
        PyObject *py_sz = PyDict_GetItemString(comp, "size");
        if (!py_sz) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'size'", ci);
            return NULL;
        }
        cs->size = (size_t)PyLong_AsUnsignedLongLong(py_sz);

        /* shape */
        PyObject *py_shape = PyDict_GetItemString(comp, "shape");
        if (!py_shape || !PyList_Check(py_shape)) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing or invalid 'shape'", ci);
            return NULL;
        }
        cs->ndim = (int)PyList_Size(py_shape);
        if (cs->ndim <= 0 || cs->ndim > 4) {
            PyErr_Format(PyExc_ValueError, "components[%zd] shape dims must be 1-4", ci);
            return NULL;
        }
        for (int d = 0; d < cs->ndim; d++) {
            cs->shape[d] = (int)PyLong_AsLong(PyList_GetItem(py_shape, d));
        }

        /* dtype — stored as string, will be passed to mx.zeros() */
        PyObject *py_dtype = PyDict_GetItemString(comp, "dtype");
        if (!py_dtype) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'dtype'", ci);
            return NULL;
        }
        const char *dtype_str = PyUnicode_AsUTF8(py_dtype);
        if (!dtype_str) return NULL;
        strncpy(cs->mx_dtype_str, dtype_str, 15);
        cs->mx_dtype_str[15] = '\0';
    }

    g_state.expert_size = expert_size;
    g_state.num_layers = num_layers;
    g_state.experts_per_layer = experts_per_layer;

    /* Open packed layer files */
    for (int li = 0; li < num_layers; li++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", packed_dir, li);

        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            PyErr_Format(PyExc_OSError, "Cannot open %s: %s", path, strerror(errno));
            /* Close already-opened fds */
            for (int k = 0; k < li; k++) {
                close(g_state.layer_fds[k].fd);
                g_state.layer_fds[k].fd = -1;
            }
            return NULL;
        }
        g_state.layer_fds[li].fd = fd;
    }

    /*
     * Import mlx and create pre-allocated arrays.
     *
     * For each (layer, slot, component), we create an mx.zeros() array
     * of the appropriate shape and dtype. Then we extract the raw CPU
     * pointer via the buffer protocol.
     *
     * We call mx.eval() on each array to ensure the Metal buffer is
     * actually allocated (lazy evaluation means the buffer might not
     * exist until evaluated).
     */

    PyObject *mx_module = PyImport_ImportModule("mlx.core");
    if (!mx_module) return NULL;

    PyObject *mx_zeros = PyObject_GetAttrString(mx_module, "zeros");
    PyObject *mx_eval = PyObject_GetAttrString(mx_module, "eval");
    if (!mx_zeros || !mx_eval) {
        Py_XDECREF(mx_zeros);
        Py_XDECREF(mx_eval);
        Py_DECREF(mx_module);
        PyErr_SetString(PyExc_RuntimeError, "Cannot find mx.zeros or mx.eval");
        return NULL;
    }

    /* Build the Python-accessible nested dict */
    PyObject *outer_dict = PyDict_New();
    if (!outer_dict) {
        Py_DECREF(mx_zeros);
        Py_DECREF(mx_eval);
        Py_DECREF(mx_module);
        return NULL;
    }

    int alloc_ok = 1;

    for (int li = 0; li < num_layers && alloc_ok; li++) {
        PyObject *layer_dict = PyDict_New();
        if (!layer_dict) { alloc_ok = 0; break; }

        for (int si = 0; si < experts_per_layer && alloc_ok; si++) {
            PyObject *slot_dict = PyDict_New();
            if (!slot_dict) { alloc_ok = 0; Py_DECREF(layer_dict); break; }

            for (int ci = 0; ci < (int)num_comps && alloc_ok; ci++) {
                ComponentSpec *cs = &g_state.comp_specs[ci];

                /* Build shape tuple */
                PyObject *py_shape = PyTuple_New(cs->ndim);
                for (int d = 0; d < cs->ndim; d++) {
                    PyTuple_SET_ITEM(py_shape, d, PyLong_FromLong(cs->shape[d]));
                }

                /* Get mx dtype object: mx.<dtype_str> */
                PyObject *mx_dtype = PyObject_GetAttrString(mx_module, cs->mx_dtype_str);
                if (!mx_dtype) {
                    PyErr_Format(PyExc_ValueError,
                                 "Unknown mlx dtype: mx.%s", cs->mx_dtype_str);
                    Py_DECREF(py_shape);
                    alloc_ok = 0;
                    Py_DECREF(slot_dict);
                    Py_DECREF(layer_dict);
                    break;
                }

                /* Call mx.zeros(shape, dtype=dtype) */
                PyObject *kwargs_dict = PyDict_New();
                PyDict_SetItemString(kwargs_dict, "dtype", mx_dtype);
                PyObject *call_args = PyTuple_Pack(1, py_shape);
                PyObject *arr = PyObject_Call(mx_zeros, call_args, kwargs_dict);

                Py_DECREF(call_args);
                Py_DECREF(kwargs_dict);
                Py_DECREF(mx_dtype);
                Py_DECREF(py_shape);

                if (!arr) {
                    alloc_ok = 0;
                    Py_DECREF(slot_dict);
                    Py_DECREF(layer_dict);
                    break;
                }

                /* mx.eval(arr) to force Metal buffer allocation */
                PyObject *eval_args = PyTuple_Pack(1, arr);
                PyObject *eval_result = PyObject_Call(mx_eval, eval_args, NULL);
                Py_DECREF(eval_args);
                Py_XDECREF(eval_result);
                if (!eval_result) {
                    Py_DECREF(arr);
                    alloc_ok = 0;
                    Py_DECREF(slot_dict);
                    Py_DECREF(layer_dict);
                    break;
                }

                /* Get the raw data pointer via buffer protocol */
                Py_buffer view;
                void *ptr = get_mx_array_ptr(arr, &view);
                if (!ptr) {
                    /*
                     * Buffer protocol failed. This can happen if MLX doesn't
                     * support writable buffers. Fall back: use np.array interface.
                     * Actually, let's try calling np.array(arr) to get a numpy
                     * array that shares the same data, then get its pointer.
                     *
                     * But first, let's try the __array_interface__ approach.
                     * MLX arrays expose __array_interface__ for numpy compat.
                     */
                    PyErr_Clear();

                    /* Try __dlpack__ / memoryview route */
                    PyObject *py_memview = PyMemoryView_FromObject(arr);
                    if (py_memview) {
                        Py_buffer *mv_buf = PyMemoryView_GET_BUFFER(py_memview);
                        ptr = mv_buf->buf;
                        /* We keep the memoryview alive via the mx.array ref */
                        Py_DECREF(py_memview);

                        /* Create a dummy view for consistency */
                        memset(&view, 0, sizeof(view));
                        view.buf = ptr;
                        view.len = (Py_ssize_t)cs->size;
                    }

                    if (!ptr) {
                        PyErr_Clear();

                        /*
                         * Last resort: use numpy to get the pointer.
                         * np.array(mx_arr, copy=False) should share memory.
                         */
                        PyObject *np_mod = PyImport_ImportModule("numpy");
                        if (np_mod) {
                            PyObject *np_asarray = PyObject_GetAttrString(np_mod, "asarray");
                            if (np_asarray) {
                                PyObject *np_arr = PyObject_CallFunctionObjArgs(np_asarray, arr, NULL);
                                if (np_arr) {
                                    if (PyObject_GetBuffer(np_arr, &view,
                                                           PyBUF_WRITABLE | PyBUF_SIMPLE) == 0) {
                                        ptr = view.buf;
                                        /* Keep np_arr alive by storing ref */
                                        /* Actually we release the buffer but keep the mx.array */
                                        PyBuffer_Release(&view);
                                    }
                                    /* Try again with read-only */
                                    if (!ptr) {
                                        PyErr_Clear();
                                        if (PyObject_GetBuffer(np_arr, &view, PyBUF_SIMPLE) == 0) {
                                            ptr = view.buf;
                                            PyBuffer_Release(&view);
                                        }
                                    }
                                    Py_DECREF(np_arr);
                                }
                                Py_DECREF(np_asarray);
                            }
                            Py_DECREF(np_mod);
                        }

                        if (!ptr) {
                            PyErr_Clear();
                            PyErr_Format(PyExc_RuntimeError,
                                         "Cannot get data pointer from mx.array for %s "
                                         "(layer=%d, slot=%d). MLX buffer protocol not available.",
                                         cs->name, li, si);
                            Py_DECREF(arr);
                            alloc_ok = 0;
                            Py_DECREF(slot_dict);
                            Py_DECREF(layer_dict);
                            break;
                        }

                        /* Store without view (pointer obtained via numpy) */
                        memset(&view, 0, sizeof(view));
                        view.buf = ptr;
                        view.len = (Py_ssize_t)cs->size;
                    }
                } else {
                    /* Got the buffer, release it but remember the pointer.
                     * The pointer remains valid as long as the mx.array lives. */
                    PyBuffer_Release(&view);
                }

                /* Store in our C-level lookup table */
                PreallocBuf *pb = &g_state.bufs[li][si][ci];
                pb->mx_array = arr;  /* We own this reference */
                pb->data_ptr = ptr;
                pb->byte_size = cs->size;

                /* Store in Python dict */
                PyObject *py_comp_key = PyUnicode_FromString(cs->name);
                PyDict_SetItem(slot_dict, py_comp_key, arr);
                Py_DECREF(py_comp_key);
                /* arr ref is held by both pb->mx_array and slot_dict */
            }

            if (alloc_ok) {
                PyObject *py_slot_key = PyLong_FromLong(si);
                PyDict_SetItem(layer_dict, py_slot_key, slot_dict);
                Py_DECREF(py_slot_key);
            }
            Py_DECREF(slot_dict);
        }

        if (alloc_ok) {
            PyObject *py_layer_key = PyLong_FromLong(li);
            PyDict_SetItem(outer_dict, py_layer_key, layer_dict);
            Py_DECREF(py_layer_key);
        }
        Py_DECREF(layer_dict);
    }

    Py_DECREF(mx_zeros);
    Py_DECREF(mx_eval);
    Py_DECREF(mx_module);

    if (!alloc_ok) {
        /* Cleanup any partially allocated arrays */
        for (int li = 0; li < num_layers; li++) {
            for (int si = 0; si < experts_per_layer; si++) {
                for (int ci = 0; ci < (int)num_comps; ci++) {
                    PreallocBuf *pb = &g_state.bufs[li][si][ci];
                    Py_XDECREF(pb->mx_array);
                    pb->mx_array = NULL;
                    pb->data_ptr = NULL;
                }
            }
        }
        /* Close fds */
        for (int li = 0; li < num_layers; li++) {
            if (g_state.layer_fds[li].fd >= 0) {
                close(g_state.layer_fds[li].fd);
                g_state.layer_fds[li].fd = -1;
            }
        }
        Py_DECREF(outer_dict);
        return NULL;
    }

    g_state.py_buffers = outer_dict;  /* We own this reference */
    g_state.preallocated = 1;

    /* Return the dict (caller gets a new reference) */
    Py_INCREF(outer_dict);
    return outer_dict;
}

/*
 * ---- load_experts(load_list) ----
 *
 * The hot path. Fills pre-allocated Metal buffers with expert weight data
 * via parallel pread() calls.
 *
 * load_list: list of (layer_idx, expert_idx, slot_idx) tuples
 *   - layer_idx: which model layer (0-based)
 *   - expert_idx: which expert in the packed file (0-based, 0-255)
 *   - slot_idx: which pre-allocated slot to fill (0-based, 0 to K-1)
 *
 * For each tuple, we pread all 9 components of the expert from the packed
 * file directly into the pre-allocated Metal buffer pointers. The GPU sees
 * the data immediately (unified memory, no copy needed).
 *
 * Returns: number of read operations completed
 */

static PyObject *fwl_load_experts(PyObject *self, PyObject *args) {
    PyObject *py_load_list = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_load_list))
        return NULL;

    if (!g_state.initialized || !g_state.preallocated) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() and prealloc() first");
        return NULL;
    }

    if (!PyList_Check(py_load_list) && !PyTuple_Check(py_load_list)) {
        PyErr_SetString(PyExc_TypeError, "load_list must be a list or tuple");
        return NULL;
    }

    Py_ssize_t num_entries = PySequence_Size(py_load_list);
    if (num_entries == 0) {
        return PyLong_FromLong(0);
    }

    /* Build work items: one per (entry, component) */
    int total_items = (int)(num_entries * g_state.num_comps);
    if (total_items > FWL_MAX_WORK_ITEMS) {
        PyErr_Format(PyExc_OverflowError,
                     "Too many work items: %d (max %d)", total_items, FWL_MAX_WORK_ITEMS);
        return NULL;
    }

    WorkItem *items = (WorkItem *)calloc(total_items, sizeof(WorkItem));
    if (!items) {
        PyErr_NoMemory();
        return NULL;
    }

    int item_idx = 0;
    for (Py_ssize_t i = 0; i < num_entries; i++) {
        PyObject *entry = PySequence_GetItem(py_load_list, i);
        if (!entry) { free(items); return NULL; }

        /* Unpack (layer_idx, expert_idx, slot_idx) */
        int layer_idx, expert_idx, slot_idx;
        if (!PyArg_ParseTuple(entry, "iii", &layer_idx, &expert_idx, &slot_idx)) {
            Py_DECREF(entry);
            free(items);
            return NULL;
        }
        Py_DECREF(entry);

        /* Validate */
        if (layer_idx < 0 || layer_idx >= g_state.num_layers) {
            PyErr_Format(PyExc_ValueError, "layer_idx %d out of range (0-%d)",
                         layer_idx, g_state.num_layers - 1);
            free(items);
            return NULL;
        }
        if (slot_idx < 0 || slot_idx >= g_state.experts_per_layer) {
            PyErr_Format(PyExc_ValueError, "slot_idx %d out of range (0-%d)",
                         slot_idx, g_state.experts_per_layer - 1);
            free(items);
            return NULL;
        }

        int fd = g_state.layer_fds[layer_idx].fd;
        off_t expert_base_offset = (off_t)expert_idx * (off_t)g_state.expert_size;

        for (int ci = 0; ci < g_state.num_comps; ci++) {
            ComponentSpec *cs = &g_state.comp_specs[ci];
            PreallocBuf *pb = &g_state.bufs[layer_idx][slot_idx][ci];

            WorkItem *wi = &items[item_idx++];
            wi->fd = fd;
            wi->dest = pb->data_ptr;
            wi->nbytes = cs->size;
            wi->file_offset = expert_base_offset + (off_t)cs->offset;
            wi->error = 0;
            wi->bytes_read = 0;
        }
    }

    /* Distribute work items to workers (round-robin) */
    /* Allocate per-worker arrays */
    WorkItem **worker_items = (WorkItem **)calloc(g_state.num_workers, sizeof(WorkItem *));
    int *worker_counts = (int *)calloc(g_state.num_workers, sizeof(int));
    if (!worker_items || !worker_counts) {
        free(items);
        free(worker_items);
        free(worker_counts);
        PyErr_NoMemory();
        return NULL;
    }

    /* Pre-count items per worker */
    int *worker_alloc = (int *)calloc(g_state.num_workers, sizeof(int));
    for (int i = 0; i < item_idx; i++) {
        worker_alloc[i % g_state.num_workers]++;
    }
    for (int w = 0; w < g_state.num_workers; w++) {
        if (worker_alloc[w] > 0) {
            worker_items[w] = (WorkItem *)calloc(worker_alloc[w], sizeof(WorkItem));
            if (!worker_items[w]) {
                for (int k = 0; k < w; k++) free(worker_items[k]);
                free(worker_items);
                free(worker_counts);
                free(worker_alloc);
                free(items);
                PyErr_NoMemory();
                return NULL;
            }
        }
    }
    free(worker_alloc);

    /* Distribute */
    for (int i = 0; i < item_idx; i++) {
        int wid = i % g_state.num_workers;
        worker_items[wid][worker_counts[wid]++] = items[i];
    }

    /* Reset completion counter */
    atomic_store(&g_state.completed_count, 0);

    /* Dispatch to workers */
    for (int w = 0; w < g_state.num_workers; w++) {
        WorkerCtx *wk = &g_state.workers[w];
        pthread_mutex_lock(&wk->work_mutex);
        wk->items = worker_items[w];
        wk->item_count = worker_counts[w];
        wk->has_work = (worker_counts[w] > 0) ? 1 : 0;
        if (wk->has_work)
            pthread_cond_signal(&wk->work_cond);
        pthread_mutex_unlock(&wk->work_mutex);
    }

    /* Release GIL and wait for all workers to complete */
    int total_expected = item_idx;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&g_state.done_mutex);
    while (atomic_load(&g_state.completed_count) < total_expected) {
        pthread_cond_wait(&g_state.done_cond, &g_state.done_mutex);
    }
    pthread_mutex_unlock(&g_state.done_mutex);
    Py_END_ALLOW_THREADS

    /* Check for errors — scan per-worker items */
    int errors = 0;
    int first_error_idx = -1;
    long long bytes_loaded = 0;
    for (int w = 0; w < g_state.num_workers; w++) {
        for (int j = 0; j < worker_counts[w]; j++) {
            if (worker_items[w][j].error != 0) {
                if (first_error_idx < 0) first_error_idx = w * 1000 + j;
                errors++;
            } else {
                bytes_loaded += worker_items[w][j].bytes_read;
            }
        }
    }

    /* Cleanup per-worker arrays */
    for (int w = 0; w < g_state.num_workers; w++) {
        free(worker_items[w]);
    }
    free(worker_items);
    free(worker_counts);
    free(items);

    /* Update stats */
    g_state.total_loads += num_entries;
    g_state.total_bytes += bytes_loaded;

    if (errors > 0) {
        PyErr_Format(PyExc_IOError,
                     "pread failed for %d/%d work items (first error at worker batch index %d)",
                     errors, item_idx, first_error_idx);
        return NULL;
    }

    return PyLong_FromLong(item_idx);
}

/*
 * ---- load_experts_coalesced(load_list) ----
 *
 * Optimized version that reads entire expert blocks in one pread per expert,
 * then splits into component buffers. Fewer syscalls = less overhead.
 *
 * Same interface as load_experts but does 1 read per expert instead of 9.
 *
 * load_list: list of (layer_idx, expert_idx, slot_idx) tuples
 */

static PyObject *fwl_load_experts_coalesced(PyObject *self, PyObject *args) {
    PyObject *py_load_list = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_load_list))
        return NULL;

    if (!g_state.initialized || !g_state.preallocated) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() and prealloc() first");
        return NULL;
    }

    if (!PyList_Check(py_load_list) && !PyTuple_Check(py_load_list)) {
        PyErr_SetString(PyExc_TypeError, "load_list must be a list or tuple");
        return NULL;
    }

    Py_ssize_t num_entries = PySequence_Size(py_load_list);
    if (num_entries == 0) {
        return PyLong_FromLong(0);
    }

    /*
     * Strategy: allocate one temp staging buffer per expert (expert_size bytes),
     * pread the entire expert block into it, then memcpy each component from
     * staging into the pre-allocated Metal buffer.
     *
     * This trades memory for fewer syscalls: 1 pread per expert vs 9.
     * For 8 experts: 8 preads vs 72.
     */

    size_t expert_sz = g_state.expert_size;

    /* Parse load list and allocate staging */
    typedef struct {
        int layer_idx;
        int expert_idx;
        int slot_idx;
        int fd;
        off_t file_offset;
        void *staging;      /* temp buffer for whole expert */
    } CoalEntry;

    CoalEntry *entries = (CoalEntry *)calloc(num_entries, sizeof(CoalEntry));
    void *staging_block = NULL;

    if (!entries) {
        PyErr_NoMemory();
        return NULL;
    }

    /* Allocate one big staging block for all experts */
    size_t total_staging = (size_t)num_entries * expert_sz;
    int rc = posix_memalign(&staging_block, FWL_PAGE_SIZE, total_staging);
    if (rc != 0 || !staging_block) {
        free(entries);
        PyErr_Format(PyExc_MemoryError,
                     "Failed to allocate %zu bytes staging", total_staging);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < num_entries; i++) {
        PyObject *entry = PySequence_GetItem(py_load_list, i);
        if (!entry) {
            free(entries);
            free(staging_block);
            return NULL;
        }

        int layer_idx, expert_idx, slot_idx;
        if (!PyArg_ParseTuple(entry, "iii", &layer_idx, &expert_idx, &slot_idx)) {
            Py_DECREF(entry);
            free(entries);
            free(staging_block);
            return NULL;
        }
        Py_DECREF(entry);

        if (layer_idx < 0 || layer_idx >= g_state.num_layers) {
            PyErr_Format(PyExc_ValueError, "layer_idx %d out of range", layer_idx);
            free(entries);
            free(staging_block);
            return NULL;
        }
        if (slot_idx < 0 || slot_idx >= g_state.experts_per_layer) {
            PyErr_Format(PyExc_ValueError, "slot_idx %d out of range", slot_idx);
            free(entries);
            free(staging_block);
            return NULL;
        }

        entries[i].layer_idx = layer_idx;
        entries[i].expert_idx = expert_idx;
        entries[i].slot_idx = slot_idx;
        entries[i].fd = g_state.layer_fds[layer_idx].fd;
        entries[i].file_offset = (off_t)expert_idx * (off_t)expert_sz;
        entries[i].staging = (char *)staging_block + (size_t)i * expert_sz;
    }

    /* Build work items: 1 pread per expert into staging */
    WorkItem *items = (WorkItem *)calloc(num_entries, sizeof(WorkItem));
    if (!items) {
        free(entries);
        free(staging_block);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < num_entries; i++) {
        items[i].fd = entries[i].fd;
        items[i].dest = entries[i].staging;
        items[i].nbytes = expert_sz;
        items[i].file_offset = entries[i].file_offset;
        items[i].error = 0;
        items[i].bytes_read = 0;
    }

    /* Distribute to workers */
    WorkItem **worker_items = (WorkItem **)calloc(g_state.num_workers, sizeof(WorkItem *));
    int *worker_counts = (int *)calloc(g_state.num_workers, sizeof(int));
    if (!worker_items || !worker_counts) {
        free(items);
        free(entries);
        free(staging_block);
        free(worker_items);
        free(worker_counts);
        PyErr_NoMemory();
        return NULL;
    }

    int *worker_alloc = (int *)calloc(g_state.num_workers, sizeof(int));
    for (Py_ssize_t i = 0; i < num_entries; i++) {
        worker_alloc[i % g_state.num_workers]++;
    }
    for (int w = 0; w < g_state.num_workers; w++) {
        if (worker_alloc[w] > 0) {
            worker_items[w] = (WorkItem *)calloc(worker_alloc[w], sizeof(WorkItem));
            if (!worker_items[w]) {
                for (int k = 0; k < w; k++) free(worker_items[k]);
                free(worker_items);
                free(worker_counts);
                free(worker_alloc);
                free(items);
                free(entries);
                free(staging_block);
                PyErr_NoMemory();
                return NULL;
            }
        }
    }
    free(worker_alloc);

    for (Py_ssize_t i = 0; i < num_entries; i++) {
        int wid = (int)(i % g_state.num_workers);
        worker_items[wid][worker_counts[wid]++] = items[i];
    }

    /* Dispatch */
    atomic_store(&g_state.completed_count, 0);

    for (int w = 0; w < g_state.num_workers; w++) {
        WorkerCtx *wk = &g_state.workers[w];
        pthread_mutex_lock(&wk->work_mutex);
        wk->items = worker_items[w];
        wk->item_count = worker_counts[w];
        wk->has_work = (worker_counts[w] > 0) ? 1 : 0;
        if (wk->has_work)
            pthread_cond_signal(&wk->work_cond);
        pthread_mutex_unlock(&wk->work_mutex);
    }

    /* Wait (without GIL) */
    int total_expected = (int)num_entries;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&g_state.done_mutex);
    while (atomic_load(&g_state.completed_count) < total_expected) {
        pthread_cond_wait(&g_state.done_cond, &g_state.done_mutex);
    }
    pthread_mutex_unlock(&g_state.done_mutex);
    Py_END_ALLOW_THREADS

    /* Check errors from staging reads */
    int errors = 0;
    for (int w = 0; w < g_state.num_workers; w++) {
        for (int j = 0; j < worker_counts[w]; j++) {
            if (worker_items[w][j].error != 0) {
                errors++;
            }
        }
    }

    if (errors > 0) {
        for (int w = 0; w < g_state.num_workers; w++) free(worker_items[w]);
        free(worker_items);
        free(worker_counts);
        free(items);
        free(entries);
        free(staging_block);
        PyErr_Format(PyExc_IOError, "pread failed for %d/%zd experts", errors, num_entries);
        return NULL;
    }

    /*
     * Now scatter from staging into the pre-allocated Metal buffers.
     * This is a pure memcpy — fast, and the Metal buffers are in unified memory
     * so the GPU sees the data immediately after.
     */
    long long bytes_loaded = 0;
    for (Py_ssize_t i = 0; i < num_entries; i++) {
        CoalEntry *ce = &entries[i];
        for (int ci = 0; ci < g_state.num_comps; ci++) {
            ComponentSpec *cs = &g_state.comp_specs[ci];
            PreallocBuf *pb = &g_state.bufs[ce->layer_idx][ce->slot_idx][ci];

            /* Copy from staging + component_offset into the Metal buffer */
            memcpy(pb->data_ptr, (char *)ce->staging + cs->offset, cs->size);
            bytes_loaded += (long long)cs->size;
        }
    }

    /* Cleanup */
    for (int w = 0; w < g_state.num_workers; w++) free(worker_items[w]);
    free(worker_items);
    free(worker_counts);
    free(items);
    free(entries);
    free(staging_block);

    /* Update stats */
    g_state.total_loads += num_entries;
    g_state.total_bytes += bytes_loaded;

    return PyLong_FromLongLong(bytes_loaded);
}

/* ---- get_buffers() — return the pre-allocated buffer dict ---- */

static PyObject *fwl_get_buffers(PyObject *self, PyObject *args) {
    if (!g_state.preallocated || !g_state.py_buffers) {
        PyErr_SetString(PyExc_RuntimeError, "No buffers pre-allocated");
        return NULL;
    }
    Py_INCREF(g_state.py_buffers);
    return g_state.py_buffers;
}

/* ---- stats() ---- */

static PyObject *fwl_stats(PyObject *self, PyObject *args) {
    return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:n, s:L, s:L}",
                         "initialized", g_state.initialized,
                         "preallocated", g_state.preallocated,
                         "num_workers", g_state.num_workers,
                         "num_layers", g_state.num_layers,
                         "experts_per_layer", g_state.experts_per_layer,
                         "expert_size", (Py_ssize_t)g_state.expert_size,
                         "total_loads", g_state.total_loads,
                         "total_bytes", g_state.total_bytes);
}

/* ---- shutdown() ---- */

static PyObject *fwl_shutdown(PyObject *self, PyObject *args) {
    if (!g_state.initialized) {
        Py_RETURN_NONE;
    }

    /* Stop worker threads */
    if (g_state.workers) {
        for (int i = 0; i < g_state.num_workers; i++) {
            WorkerCtx *w = &g_state.workers[i];
            pthread_mutex_lock(&w->work_mutex);
            w->running = 0;
            w->has_work = 1;
            pthread_cond_signal(&w->work_cond);
            pthread_mutex_unlock(&w->work_mutex);
        }
        for (int i = 0; i < g_state.num_workers; i++) {
            pthread_join(g_state.workers[i].thread, NULL);
            pthread_mutex_destroy(&g_state.workers[i].work_mutex);
            pthread_cond_destroy(&g_state.workers[i].work_cond);
        }
        free(g_state.workers);
        g_state.workers = NULL;
    }

    /* Release pre-allocated mx.array references */
    if (g_state.preallocated) {
        for (int li = 0; li < g_state.num_layers; li++) {
            for (int si = 0; si < g_state.experts_per_layer; si++) {
                for (int ci = 0; ci < g_state.num_comps; ci++) {
                    PreallocBuf *pb = &g_state.bufs[li][si][ci];
                    Py_XDECREF(pb->mx_array);
                    pb->mx_array = NULL;
                    pb->data_ptr = NULL;
                }
            }
        }
        Py_XDECREF(g_state.py_buffers);
        g_state.py_buffers = NULL;
        g_state.preallocated = 0;
    }

    /* Close layer file descriptors */
    for (int i = 0; i < g_state.num_layers; i++) {
        if (g_state.layer_fds[i].fd >= 0) {
            close(g_state.layer_fds[i].fd);
            g_state.layer_fds[i].fd = -1;
        }
    }
    g_state.num_layers = 0;

    /* Destroy sync primitives */
    pthread_mutex_destroy(&g_state.done_mutex);
    pthread_cond_destroy(&g_state.done_cond);

    g_state.initialized = 0;
    g_state.num_workers = 0;
    g_state.num_comps = 0;
    g_state.expert_size = 0;
    g_state.experts_per_layer = 0;
    g_state.total_loads = 0;
    g_state.total_bytes = 0;

    Py_RETURN_NONE;
}

/* ---- Module definition ---- */

static PyMethodDef fwl_methods[] = {
    {"init", (PyCFunction)fwl_init, METH_VARARGS | METH_KEYWORDS,
     "init(num_workers=8) -- Create persistent worker thread pool for parallel pread"},
    {"prealloc", (PyCFunction)fwl_prealloc, METH_VARARGS | METH_KEYWORDS,
     "prealloc(num_layers, experts_per_layer, components, packed_dir, expert_size=0)\n"
     "Pre-allocate mx.array Metal buffers and open packed layer files.\n"
     "Returns: dict[layer][slot][comp_name] = mx.array"},
    {"load_experts", fwl_load_experts, METH_VARARGS,
     "load_experts(load_list)\n"
     "Fill pre-allocated Metal buffers via parallel pread (9 reads per expert).\n"
     "load_list: [(layer_idx, expert_idx, slot_idx), ...]\n"
     "Returns: number of read operations completed"},
    {"load_experts_coalesced", fwl_load_experts_coalesced, METH_VARARGS,
     "load_experts_coalesced(load_list)\n"
     "Fill pre-allocated Metal buffers via coalesced pread (1 read per expert).\n"
     "Reads entire expert block to staging, then memcpy to Metal buffers.\n"
     "load_list: [(layer_idx, expert_idx, slot_idx), ...]\n"
     "Returns: total bytes loaded"},
    {"get_buffers", fwl_get_buffers, METH_NOARGS,
     "get_buffers() -- Return the pre-allocated buffer dict"},
    {"stats", fwl_stats, METH_NOARGS,
     "stats() -- Return diagnostic counters"},
    {"shutdown", fwl_shutdown, METH_NOARGS,
     "shutdown() -- Stop workers, release buffers, close files"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fwl_module = {
    PyModuleDef_HEAD_INIT,
    "fast_weight_load",
    "Zero-copy expert weight loading into pre-allocated MLX Metal buffers.\n"
    "\n"
    "Eliminates np.frombuffer + mx.array() overhead by:\n"
    "1. Pre-allocating mx.array objects (Metal-backed unified memory)\n"
    "2. Getting raw CPU pointers via Python buffer protocol\n"
    "3. Using pread() directly into Metal buffers from pthreads (no GIL)\n"
    "\n"
    "On Apple Silicon unified memory, pread into a Metal buffer pointer\n"
    "makes data immediately visible to the GPU with no upload step.",
    -1,
    fwl_methods
};

PyMODINIT_FUNC PyInit_fast_weight_load(void) {
    PyObject *m = PyModule_Create(&fwl_module);
    if (!m) return NULL;

    PyModule_AddIntConstant(m, "MAX_LAYERS", FWL_MAX_LAYERS);
    PyModule_AddIntConstant(m, "MAX_SLOTS", FWL_MAX_SLOTS);
    PyModule_AddIntConstant(m, "MAX_COMPONENTS", FWL_MAX_COMPONENTS);
    PyModule_AddIntConstant(m, "MAX_WORKERS", FWL_MAX_WORKERS);
    PyModule_AddIntConstant(m, "PAGE_SIZE", FWL_PAGE_SIZE);

    return m;
}
