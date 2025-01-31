#ifndef ZEPHYR_XNNPACK_PTHREAD_H
#define ZEPHYR_XNNPACK_PTHREAD_H

#include <stdint.h>
#include <stdbool.h>

// Minimal typedefs and macros to satisfy XNNPack
#define PTHREAD_ONCE_INIT {0}

typedef uint32_t pthread_mutex_t;
typedef uint32_t pthread_t;
typedef uint32_t pthread_cond_t;

struct pthread_once {
	bool flag;
};
typedef struct pthread_once pthread_once_t;

// Define empty function signatures so XNNPack compiles
// static inline int pthread_mutex_init(pthread_mutex_t *mutex, void *attr) { return 0; }
// static inline int pthread_mutex_lock(pthread_mutex_t *mutex) { return 0; }
// static inline int pthread_mutex_unlock(pthread_mutex_t *mutex) { return 0; }
// static inline int pthread_mutex_destroy(pthread_mutex_t *mutex) { return 0; }
// static int sched_yield 	( 	void		) 	{return 0;}
// static long sysconf 	( 	int 	opt	) {return 4096L;}	

extern int pthread_mutex_init(pthread_mutex_t *mutex, void *attr);
extern int pthread_mutex_lock(pthread_mutex_t *mutex);
extern int pthread_mutex_unlock(pthread_mutex_t *mutex);
extern int pthread_mutex_destroy(pthread_mutex_t *mutex);

// extern int sched_yield(void);
extern long sysconf(int opt);

extern void z_impl_k_yield(void);

#define compiler_barrier() do { \
	__asm__ __volatile__ ("" ::: "memory"); \
} while (false)

static inline void k_yield(void)
{
	compiler_barrier();
	z_impl_k_yield();
}

static inline int sched_yield(void)
{
	k_yield();
	return 0;
}




#endif // ZEPHYR_XNNPACK_PTHREAD_H
