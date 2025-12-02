/**
 * Lock-Free Ring Buffer - Standalone System Integration Component
 * Optimized for high-throughput inter-system communication
 * Single Producer Single Consumer (SPSC) implementation
 */

#ifndef LOCKFREE_RINGBUFFER_H
#define LOCKFREE_RINGBUFFER_H

#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define CACHE_LINE_SIZE 64
#define RING_BUFFER_ALIGNMENT __attribute__((aligned(CACHE_LINE_SIZE)))

typedef enum {
    RING_SUCCESS = 0,
    RING_FULL = 1,
    RING_EMPTY = 2,
    RING_INVALID = 3
} ring_result_t;

typedef struct {
    uint64_t total_writes;
    uint64_t total_reads;
    uint64_t failed_writes;
    uint32_t peak_usage;
    uint32_t current_usage;
} ring_metrics_t;

typedef struct {
    char* data;
    uint32_t capacity;
    uint32_t item_size;
    uint32_t mask;
    
    RING_BUFFER_ALIGNMENT atomic_uint32_t write_idx;
    RING_BUFFER_ALIGNMENT atomic_uint32_t read_idx;
    RING_BUFFER_ALIGNMENT ring_metrics_t metrics;
} ring_buffer_t;

// Core API - Essential functions only
ring_buffer_t* ring_create(uint32_t capacity, uint32_t item_size);
void ring_destroy(ring_buffer_t* ring);
ring_result_t ring_write(ring_buffer_t* ring, const void* data);
ring_result_t ring_read(ring_buffer_t* ring, void* data);
uint32_t ring_size(const ring_buffer_t* ring);
bool ring_is_empty(const ring_buffer_t* ring);
bool ring_is_full(const ring_buffer_t* ring);
void ring_get_metrics(const ring_buffer_t* ring, ring_metrics_t* metrics);

#endif // LOCKFREE_RINGBUFFER_H

/* Implementation */

static uint32_t next_power_of_two(uint32_t n) {
    n--;
    n |= n >> 1;  n |= n >> 2;  n |= n >> 4;
    n |= n >> 8;  n |= n >> 16;
    return ++n;
}

ring_buffer_t* ring_create(uint32_t capacity, uint32_t item_size) {
    if (capacity < 2 || item_size == 0) return NULL;
    
    capacity = next_power_of_two(capacity);
    ring_buffer_t* ring = aligned_alloc(CACHE_LINE_SIZE, sizeof(ring_buffer_t));
    if (!ring) return NULL;
    
    ring->data = aligned_alloc(CACHE_LINE_SIZE, capacity * item_size);
    if (!ring->data) {
        free(ring);
        return NULL;
    }
    
    ring->capacity = capacity;
    ring->item_size = item_size;
    ring->mask = capacity - 1;
    
    atomic_init(&ring->write_idx, 0);
    atomic_init(&ring->read_idx, 0);
    memset(&ring->metrics, 0, sizeof(ring_metrics_t));
    
    return ring;
}

void ring_destroy(ring_buffer_t* ring) {
    if (ring) {
        free(ring->data);
        free(ring);
    }
}

ring_result_t ring_write(ring_buffer_t* ring, const void* data) {
    if (!ring || !data) return RING_INVALID;
    
    uint32_t write_pos = atomic_load_explicit(&ring->write_idx, memory_order_relaxed);
    uint32_t next_write = (write_pos + 1) & ring->mask;
    uint32_t read_pos = atomic_load_explicit(&ring->read_idx, memory_order_acquire);
    
    if (next_write == read_pos) {
        ring->metrics.failed_writes++;
        return RING_FULL;
    }
    
    // Write data
    char* slot = ring->data + (write_pos * ring->item_size);
    memcpy(slot, data, ring->item_size);
    
    // Update write index with release semantics
    atomic_store_explicit(&ring->write_idx, next_write, memory_order_release);
    
    // Update metrics
    ring->metrics.total_writes++;
    uint32_t usage = ring_size(ring);
    if (usage > ring->metrics.peak_usage) {
        ring->metrics.peak_usage = usage;
    }
    ring->metrics.current_usage = usage;
    
    return RING_SUCCESS;
}

ring_result_t ring_read(ring_buffer_t* ring, void* data) {
    if (!ring || !data) return RING_INVALID;
    
    uint32_t read_pos = atomic_load_explicit(&ring->read_idx, memory_order_relaxed);
    uint32_t write_pos = atomic_load_explicit(&ring->write_idx, memory_order_acquire);
    
    if (read_pos == write_pos) {
        return RING_EMPTY;
    }
    
    // Read data
    char* slot = ring->data + (read_pos * ring->item_size);
    memcpy(data, slot, ring->item_size);
    
    // Update read index with release semantics
    uint32_t next_read = (read_pos + 1) & ring->mask;
    atomic_store_explicit(&ring->read_idx, next_read, memory_order_release);
    
    // Update metrics
    ring->metrics.total_reads++;
    ring->metrics.current_usage = ring_size(ring);
    
    return RING_SUCCESS;
}

uint32_t ring_size(const ring_buffer_t* ring) {
    if (!ring) return 0;
    
    uint32_t write_pos = atomic_load_explicit(&ring->write_idx, memory_order_acquire);
    uint32_t read_pos = atomic_load_explicit(&ring->read_idx, memory_order_acquire);
    
    return (write_pos - read_pos) & ring->mask;
}

bool ring_is_empty(const ring_buffer_t* ring) {
    if (!ring) return true;
    
    uint32_t write_pos = atomic_load_explicit(&ring->write_idx, memory_order_acquire);
    uint32_t read_pos = atomic_load_explicit(&ring->read_idx, memory_order_acquire);
    
    return read_pos == write_pos;
}

bool ring_is_full(const ring_buffer_t* ring) {
    if (!ring) return false;
    
    uint32_t write_pos = atomic_load_explicit(&ring->write_idx, memory_order_relaxed);
    uint32_t next_write = (write_pos + 1) & ring->mask;
    uint32_t read_pos = atomic_load_explicit(&ring->read_idx, memory_order_acquire);
    
    return next_write == read_pos;
}

void ring_get_metrics(const ring_buffer_t* ring, ring_metrics_t* metrics) {
    if (!ring || !metrics) return;
    
    *metrics = ring->metrics;
    metrics->current_usage = ring_size(ring);
}

/* Usage Example - System Integration */
/*

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

typedef struct {
    uint64_t timestamp;
    uint32_t system_id;
    uint32_t severity;
    char message[56]; // Total struct = 72 bytes (cache-friendly)
} log_entry_t;

// Producer system
void* logger_system(void* arg) {
    ring_buffer_t* ring = (ring_buffer_t*)arg;
    log_entry_t entry;
    
    for (int i = 0; i < 1000; i++) {
        entry.timestamp = i;
        entry.system_id = 1;
        entry.severity = i % 4;
        snprintf(entry.message, sizeof(entry.message), "Log message %d", i);
        
        while (ring_write(ring, &entry) == RING_FULL) {
            usleep(100); // Backpressure handling
        }
    }
    return NULL;
}

// Consumer system
void* metrics_system(void* arg) {
    ring_buffer_t* ring = (ring_buffer_t*)arg;
    log_entry_t entry;
    
    while (1) {
        if (ring_read(ring, &entry) == RING_SUCCESS) {
            // Process log entry for metrics
            printf("Processed: %s\n", entry.message);
        } else {
            usleep(1000);
        }
    }
    return NULL;
}

int main() {
    ring_buffer_t* shared_ring = ring_create(1024, sizeof(log_entry_t));
    
    pthread_t producer, consumer;
    pthread_create(&producer, NULL, logger_system, shared_ring);
    pthread_create(&consumer, NULL, metrics_system, shared_ring);
    
    pthread_join(producer, NULL);
    pthread_cancel(consumer);
    
    ring_metrics_t metrics;
    ring_get_metrics(shared_ring, &metrics);
    printf("Total writes: %lu, reads: %lu, peak usage: %u\n", 
           metrics.total_writes, metrics.total_reads, metrics.peak_usage);
    
    ring_destroy(shared_ring);
    return 0;
}

*/