#pragma once

#include <stdlib.h>
#include <stdatomic.h>

typedef float dist_t;

// a point in world coordinates
typedef struct {
    dist_t x;
    dist_t y;
} point_t;

typedef point_t vec_t;

static inline vec_t point_diff(point_t a, point_t b) {
    return (vec_t){a.x - b.x, a.y - b.y};
}

static inline point_t point_add_scaled_vec(point_t a, vec_t b, float scale) {
    return (point_t){a.x + b.x * scale, a.y + b.y * scale};
}

// a circle in world coordinates
typedef struct {
    point_t center;
    dist_t radius;
    float shade;
} circle_t;

// data needed to render a frame in world coordinates
typedef struct {
    size_t num_circles;
    circle_t circles[];
} frame_t;

typedef enum {
    INPUT_TYPE_FORCE_POINT,
    INPUT_TYPE_STOP_FORCE,
    INPUT_TYPE_SPAWN,
    INPUT_TYPE_COMMAND,
} input_type_t;

typedef struct {
    input_type_t type; // INPUT_TYPE_FORCE_POINT
    union input *next;
    point_t point;
    enum {
        FORCE_TYPE_GRAVITY,
        FORCE_TYPE_COULOMB,
    } force_type;
    float strength;
} input_force_point_t;

typedef struct {
    input_type_t type; // INPUT_TYPE_SPAWN
    union input *next;
    point_t point;
    float mass;
    float charge;
} input_spawn_t;

typedef struct {
    input_type_t type; // INPUT_TYPE_COMMAND
    union input *next;
    enum {
        COMMAND_TYPE_PAUSE,
        COMMAND_TYPE_RESUME,
        COMMAND_TYPE_RESET,
    } command_type;
} input_command_t;

typedef union input {
    struct {
        input_type_t type;
        union input *next;
    } base;
    input_force_point_t force_point;
    input_spawn_t spawn;
    input_command_t command;
} input_t;

typedef struct {
    frame_t * _Atomic frame;
    input_t * _Atomic input;
} shared_data_t;