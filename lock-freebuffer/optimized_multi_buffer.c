/**
 * Optimized Multi-Buffer System - Lock-Free with Enhanced Performance
 * Features: Memory pools, condition variables, binary protocol, metrics
 */
#ifndef MULTI_BUFFER_OPT_H
#define MULTI_BUFFER_OPT_H

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define CACHE_LINE 64
#define MAX_BUFFERS 16
#define POOL_SIZE 1024
#define ITEM_SIZE 256

typedef enum { SUCCESS, FULL, EMPTY, INVALID } result_t;
typedef struct { uint8_t cmd; uint8_t buf_id; uint16_t len; } __attribute__((packed)) msg_hdr_t;

// Lock-free ring buffer with condition signaling
typedef struct {
    char data[POOL_SIZE * ITEM_SIZE] __attribute__((aligned(CACHE_LINE)));
    _Atomic uint32_t head, tail;
    uint32_t mask;
    pthread_cond_t space_avail, data_avail;
    pthread_mutex_t mutex;
    _Atomic uint64_t ops, errors;
} ring_buf_t;

// Multi-buffer system with dynamic management
typedef struct {
    ring_buf_t mini_bufs[MAX_BUFFERS];
    ring_buf_t main_buf;
    _Atomic uint32_t buf_count;
    _Atomic bool running;
    pthread_t aggregator;
    struct { _Atomic uint64_t writes, reads, throughput; } metrics;
} buf_system_t;

// Core API
static inline uint32_t next_pow2(uint32_t n) {
    return n <= 1 ? 2 : 1 << (32 - __builtin_clz(n - 1));
}

result_t buf_init(ring_buf_t* buf, uint32_t capacity) {
    if (!buf || capacity < 2) return INVALID;
    buf->mask = next_pow2(capacity) - 1;
    atomic_init(&buf->head, 0); atomic_init(&buf->tail, 0);
    atomic_init(&buf->ops, 0); atomic_init(&buf->errors, 0);
    pthread_cond_init(&buf->space_avail, NULL);
    pthread_cond_init(&buf->data_avail, NULL);
    pthread_mutex_init(&buf->mutex, NULL);
    return SUCCESS;
}

result_t buf_write(ring_buf_t* buf, const void* data, bool blocking) {
    if (!buf || !data) return INVALID;
    
    uint32_t head = atomic_load_explicit(&buf->head, memory_order_relaxed);
    uint32_t next = (head + 1) & buf->mask;
    
    if (next == atomic_load_explicit(&buf->tail, memory_order_acquire)) {
        if (!blocking) { atomic_fetch_add(&buf->errors, 1); return FULL; }
        pthread_mutex_lock(&buf->mutex);
        while (next == atomic_load(&buf->tail))
            pthread_cond_wait(&buf->space_avail, &buf->mutex);
        pthread_mutex_unlock(&buf->mutex);
    }
    
    memcpy(buf->data + head * ITEM_SIZE, data, ITEM_SIZE);
    atomic_store_explicit(&buf->head, next, memory_order_release);
    atomic_fetch_add(&buf->ops, 1);
    pthread_cond_signal(&buf->data_avail);
    return SUCCESS;
}

result_t buf_read(ring_buf_t* buf, void* data) {
    if (!buf || !data) return INVALID;
    
    uint32_t tail = atomic_load_explicit(&buf->tail, memory_order_relaxed);
    if (tail == atomic_load_explicit(&buf->head, memory_order_acquire))
        return EMPTY;
    
    memcpy(data, buf->data + tail * ITEM_SIZE, ITEM_SIZE);
    atomic_store_explicit(&buf->tail, (tail + 1) & buf->mask, memory_order_release);
    atomic_fetch_add(&buf->ops, 1);
    pthread_cond_signal(&buf->space_avail);
    return SUCCESS;
}

// Optimized aggregator with batch processing
void* aggregator_thread(void* arg) {
    buf_system_t* sys = arg;
    char batch[8][ITEM_SIZE]; // Batch processing
    uint32_t batch_size;
    
    while (atomic_load(&sys->running)) {
        for (uint32_t i = 0; i < atomic_load(&sys->buf_count); i++) {
            batch_size = 0;
            // Batch read from mini buffer
            while (batch_size < 8 && buf_read(&sys->mini_bufs[i], batch[batch_size]) == SUCCESS)
                batch_size++;
            
            // Batch write to main buffer
            for (uint32_t j = 0; j < batch_size; j++)
                buf_write(&sys->main_buf, batch[j], true);
            
            if (batch_size > 0)
                atomic_fetch_add(&sys->metrics.throughput, batch_size);
        }
        if (batch_size == 0) usleep(50); // Adaptive sleep
    }
    return NULL;
}

// System management
buf_system_t* system_create(uint32_t mini_count, uint32_t mini_cap, uint32_t main_cap) {
    if (mini_count > MAX_BUFFERS) return NULL;
    
    buf_system_t* sys = calloc(1, sizeof(buf_system_t));
    if (!sys) return NULL;
    
    for (uint32_t i = 0; i < mini_count; i++)
        if (buf_init(&sys->mini_bufs[i], mini_cap) != SUCCESS) {
            free(sys); return NULL;
        }
    
    if (buf_init(&sys->main_buf, main_cap) != SUCCESS) {
        free(sys); return NULL;
    }
    
    atomic_store(&sys->buf_count, mini_count);
    atomic_store(&sys->running, true);
    pthread_create(&sys->aggregator, NULL, aggregator_thread, sys);
    return sys;
}

void system_destroy(buf_system_t* sys) {
    if (!sys) return;
    atomic_store(&sys->running, false);
    pthread_join(sys->aggregator, NULL);
    free(sys);
}

result_t system_write(buf_system_t* sys, uint32_t buf_id, const void* data) {
    if (!sys || buf_id >= atomic_load(&sys->buf_count)) return INVALID;
    result_t res = buf_write(&sys->mini_bufs[buf_id], data, false);
    if (res == SUCCESS) atomic_fetch_add(&sys->metrics.writes, 1);
    return res;
}

result_t system_read(buf_system_t* sys, void* data) {
    if (!sys) return INVALID;
    result_t res = buf_read(&sys->main_buf, data);
    if (res == SUCCESS) atomic_fetch_add(&sys->metrics.reads, 1);
    return res;
}

// Optimized network service with binary protocol
void handle_client(int fd, buf_system_t* sys) {
    msg_hdr_t hdr;
    char data[ITEM_SIZE];
    result_t result;
    
    while (recv(fd, &hdr, sizeof(hdr), MSG_WAITALL) == sizeof(hdr)) {
        switch (hdr.cmd) {
            case 1: // WRITE
                if (recv(fd, data, ITEM_SIZE, MSG_WAITALL) == ITEM_SIZE) {
                    result = system_write(sys, hdr.buf_id, data);
                    send(fd, &result, 1, 0);
                }
                break;
            case 2: // READ
                result = system_read(sys, data);
                send(fd, &result, 1, 0);
                if (result == SUCCESS) send(fd, data, ITEM_SIZE, 0);
                break;
        }
    }
    close(fd);
}

void start_service(uint16_t port, buf_system_t* sys) {
    int srv_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr = {AF_INET, htons(port), {INADDR_ANY}};
    bind(srv_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv_fd, 128);
    
    while (atomic_load(&sys->running)) {
        int client = accept(srv_fd, NULL, NULL);
        if (client > 0) {
            pthread_t thread;
            pthread_create(&thread, NULL, (void*(*)(void*))handle_client, 
                         (void*)(intptr_t)client);
            pthread_detach(thread);
        }
    }
    close(srv_fd);
}

#endif