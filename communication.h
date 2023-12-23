#pragma once

#include <stdlib.h>
#include <stdatomic.h>

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
    INPUT_TYPE_COMMAND,
} input_type_t;

typedef struct {
    input_type_t type; // INPUT_TYPE_FORCE_POINT
    point_t point;
    enum {
        FORCE_TYPE_GRAVITY,
        FORCE_TYPE_COULOMB,
    } force_type;
    float strength;
} input_force_point_t;

typedef struct {
    input_type_t type; // INPUT_TYPE_SPAWN
    point_t point;
    float mass;
    float charge;
} input_spawn_t;

typedef struct {
    input_type_t type; // INPUT_TYPE_COMMAND
    enum {
        COMMAND_TYPE_PAUSE,
        COMMAND_TYPE_RESUME,
        COMMAND_TYPE_RESET,
    } command_type;
} input_command_t;

typedef union {
    input_type_t type;
    input_force_point_t force_point;
    input_spawn_t spawn;
    input_command_t command;
} input_t;

typedef struct {
    frame_t * _Atomic frame;
    input_t * _Atomic input;
} shared_data_t;