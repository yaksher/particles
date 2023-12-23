#pragma once

#include <stdlib.h>

// a point in world coordinates
typedef struct {
    float x;
    float y;
} point_t;

typedef point_t vec_t;

// a circle in world coordinates
typedef struct {
    point_t center;
    float radius;
    float shade;
} circle_t;

// data needed to render a frame in world coordinates
typedef struct {
    size_t num_circles;
    circle_t circles[];
} frame_t;

typedef enum {
    INPUT_TYPE_FORCE_POINT,
    INPUT_TYPE_SPAWN,
} input_type_t;

typedef union {
    struct {
        input_type_t type;
        point_t point;
        enum {
            FORCE_TYPE_GRAVITY,
            FORCE_TYPE_COULOMB,
        } force_type;
    } force_point;
    struct {
        input_type_t type;
        point_t point;
        float mass;
        float charge;
    } spawn;
} input_t;