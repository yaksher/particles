#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>

#include "simulation.h"

#define BILLION 1000000000L

#define NUM_WORKERS 32

typedef double world_time_t;

typedef struct {
    float mass;
    float charge;
    float clustering;
    dist_t radius;
    bool fixed;
} particle_t;

typedef struct {
    point_t pos;
    vec_t vel;
} phase_pair_t;

typedef struct {
    size_t num_particles;
    particle_t *particles;
    phase_pair_t *phases;
    bool paused;
    double gravity;
    double coulomb;
    double collision;
    double clustering;
    double global_clustering;
} world_t;

typedef struct {
    world_t *world;
    world_time_t dt;
    float max_force;
    phase_pair_t *next;
    frame_t *next_frame;
    size_t num_done;
    uint64_t should_start_mask;
    pthread_cond_t main;
    pthread_mutex_t main_mutex;
    pthread_cond_t workers;
    pthread_mutex_t workers_mutex;
} step_helper_data_t;

// ensures that a bit-mask for the works fits inside the should_start_mask
#define BYTE_BITS 8
#define MAX_NUM_WORKERS ((int) sizeof(((step_helper_data_t *) NULL)->should_start_mask) * BYTE_BITS)
typedef int assert_max_num_workers[MAX_NUM_WORKERS - NUM_WORKERS];

typedef struct {
    step_helper_data_t *data;
    size_t worker_id;
} step_helper_arg_t;

double clamp(double x, double min, double max) {
    if (x < min) {
        return min;
    } else if (x > max) {
        return max;
    } else {
        return x;
    }
}

static double unif_rand() {
    return rand() / (double) RAND_MAX;
}

static double norm_rand() {
    const double EPSILON = 1e-12;
    double u;
    do {
        u = unif_rand();
    } while (u <= EPSILON);
    double v = unif_rand();
    return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

static float radius_of_mass(float mass) {
    return sqrt(mass);
}

#define CHARGE_RANGE 5
static inline float particle_shade(particle_t *particle, phase_pair_t *phase) {
    // return (particle->charge + CHARGE_RANGE) / (2 * CHARGE_RANGE);
    (void) particle;
    dist_t speed = sqrtf(phase->vel.x * phase->vel.x + phase->vel.y * phase->vel.y);
    return 1 - expf(-speed / 300);
}

// compute the frame from the world state
static frame_t *make_frame(world_t *world) {
    frame_t *frame = malloc(sizeof(frame_t) + world->num_particles * sizeof(circle_t));
    frame->num_circles = world->num_particles;
    circle_t *circles = frame->circles;
    for (size_t i = 0; i < frame->num_circles; i++) {
        circles[i].center.x = world->phases[i].pos.x;
        circles[i].center.y = world->phases[i].pos.y;
        circles[i].radius = world->particles[i].radius * 1.2;
        circles[i].shade = particle_shade(&world->particles[i], &world->phases[i]);
    }
    return frame;
}

static void init_frame(frame_t *frame, world_t *world, size_t first_i, size_t last_i) {
    circle_t *circles = frame->circles;
    for (size_t i = first_i; i < last_i; i++) {
        circles[i].center.x = world->phases[i].pos.x;
        circles[i].center.y = world->phases[i].pos.y;
        circles[i].radius = world->particles[i].radius * 1.2;
        circles[i].shade = particle_shade(&world->particles[i], &world->phases[i]);
    }
}

// wait until the next tick
// if the last tick took too long, return the time that needs to be
// made up
static uint64_t wait_tick(struct timespec *last_tick, uint64_t target_nsecs) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t elapsed_nsecs = (now.tv_sec - last_tick->tv_sec) * BILLION 
                            + (now.tv_nsec - last_tick->tv_nsec);

    static double mean_tick_time = 0;
    mean_tick_time = 0.9 * mean_tick_time + 0.1 * (double) elapsed_nsecs / BILLION;
    fprintf(stderr, "\rMean tick time %0.5fs.", mean_tick_time);
    if (elapsed_nsecs > BILLION) {
        fprintf(stderr, "warning: frame rate below 1 fps\n");
        *last_tick = now;
        return 0;
    } else if (elapsed_nsecs < target_nsecs) {
        struct timespec rem = {
            .tv_sec = 0,
            .tv_nsec = target_nsecs - elapsed_nsecs
        };
        nanosleep(&rem, NULL);
        *last_tick = now;
        return 0;
    } else {
        *last_tick = now;
        return elapsed_nsecs - target_nsecs;
    }
}

// FEATURES:
#define GRAVITY
// #define COULOMB
#define COLLISION
// #define CLUSTERING
// #define GLOBAL_CLUSTERING

// initialize the world; does not affect pause value (which thus should be initialized
// separately before or after using this)
static void init_world(world_t *world) {
    const size_t NUM_PARTICLES = 5000;
    bool paused = world->paused;
    *world = (world_t) {
        .paused = paused,
        .gravity = 1000,
        .coulomb = 50,
        .collision = 10,
        .clustering = 10,
        .global_clustering = 0.5 / NUM_PARTICLES,
        .num_particles = NUM_PARTICLES,
        .particles = malloc(NUM_PARTICLES * sizeof(particle_t)),
        .phases = malloc(NUM_PARTICLES * sizeof(phase_pair_t))
    };
    world->phases[0].pos.x = 0;
    world->phases[0].pos.y = 0;
    world->phases[0].vel.x = 0;
    world->phases[0].vel.y = 0;
    world->particles[0].mass = 100000;
    world->particles[0].charge = 0;
    world->particles[0].clustering = 0;
    world->particles[0].radius = 50;
    world->particles[0].fixed = true;
    const dist_t RING_RADIUS = 2000;
    dist_t velocity = 0.3 * sqrtf(world->gravity * world->particles[0].mass);
    for (size_t i = 1; i < world->num_particles; i++) {
        // Init position:
        // pick a random direction
        vec_t dir = {
            .x = norm_rand(),
            .y = norm_rand()
        };
        // get the inverse radius
        float inv_rad = sqrtf(dir.x * dir.x + dir.y * dir.y);
        // ensure the radius isn't 0
        inv_rad = fmaxf(inv_rad, 0.001);
        dist_t rad = RING_RADIUS / inv_rad;
        // add some noise
        if (i < NUM_PARTICLES * 0.3) {
            rad *= 0.2;
        }
        rad *= norm_rand() * 0.03 + 1;
        phase_pair_t *phase = &world->phases[i];
        // scale the direction vector to the desired radius
        phase->pos.x = dir.x * rad;
        phase->pos.y = dir.y * rad;

        // Init velocity (approx. tangential to the ring):
        if (i < NUM_PARTICLES * 0.3) {
            velocity *= 0.01;
        }
        phase->vel.x = phase->pos.y / sqrtf(rad) / rad * velocity + norm_rand() * 0.;
        phase->vel.y = -phase->pos.x / sqrtf(rad) / rad * velocity + norm_rand() * 0.;
        if (i < NUM_PARTICLES * 0.3) {
            velocity *= 100;
        }

        // Init mass, charge, clustering, and radius:
        // particle.radius = radius_of_mass(particle.mass);
        particle_t particle = {
            .mass = fmax(10 + norm_rand() * 5, 0.01),
            .radius = radius_of_mass(particle.mass),
            .charge = unif_rand() < 0.5 ? CHARGE_RANGE : -CHARGE_RANGE,
            .clustering = 2,
            .fixed = false,
        };
        // particle.charge = clamp(norm_rand() * 5, -CHARGE_RANGE, CHARGE_RANGE);
        // particle.charge = unif_rand() * 2 * CHARGE_RANGE - CHARGE_RANGE;
        world->particles[i] = particle;
    }
}

static frame_t *step(step_helper_data_t *data, world_time_t dt, float max_force);
static void *step_helper(void *);

static void process_input(input_t *input, world_t *world, world_time_t dt) {
    if (input == NULL) {
        return;
    }
    static bool force_point_active = false;
    static input_force_point_t force_point;
    while (input) {
        switch (input->base.type) {
            case INPUT_TYPE_FORCE_POINT: {
                force_point = input->force_point;
                force_point_active = true;
                break;
            }
            case INPUT_TYPE_STOP_FORCE: {
                force_point_active = false;
                break;
            }
            case INPUT_TYPE_SPAWN: {
                input_spawn_t *spawn = &input->spawn;
                world->particles = realloc(world->particles, (world->num_particles + 1) * sizeof(particle_t));
                world->phases = realloc(world->phases, (world->num_particles + 1) * sizeof(phase_pair_t));
                particle_t particle;
                phase_pair_t phase;
                phase.pos.x = spawn->point.x;
                phase.pos.y = spawn->point.y;
                phase.vel.x = 0;
                phase.vel.y = 0;
                particle.mass = spawn->mass;
                particle.charge = spawn->charge;
                particle.clustering = 1;
                particle.radius = radius_of_mass(particle.mass);
                world->particles[world->num_particles] = particle;
                world->phases[world->num_particles] = phase;
                world->num_particles++;
                break;
            }
            case INPUT_TYPE_COMMAND: {
                input_command_t *command = &input->command;
                switch (command->command_type) {
                    case COMMAND_TYPE_PAUSE:
                        world->paused = true;
                        break;
                    case COMMAND_TYPE_RESUME:
                        world->paused = false;
                        break;
                    case COMMAND_TYPE_RESET:
                        free(world->particles);
                        free(world->phases);
                        init_world(world);
                        break;
                }
                break;
            }
            default:
                fprintf(stderr, "warning: unknown input type %d\n", input->base.type);
        }
        input_t *cur = input;
        input = input->base.next;
        free(cur);
    }
    for (size_t i = 0; force_point_active && i < world->num_particles; i++) {
        dist_t dx = world->phases[i].pos.x - force_point.point.x;
        dist_t dy = world->phases[i].pos.y - force_point.point.y;
        float dist_sq = dx * dx + dy * dy;
        dist_t dist = sqrtf(dist_sq);
        float force = 0;
        switch (force_point.force_type) {
            case FORCE_TYPE_GRAVITY:
                force = world->gravity * world->particles[i].mass * force_point.strength / dist_sq;
                break;
            case FORCE_TYPE_COULOMB:
                force = world->coulomb * world->particles[i].charge * force_point.strength / dist_sq;
                break;
        }
        force *= dt;
        vec_t force_vec = {
            .x = force * dx / dist,
            .y = force * dy / dist
        };
        world->phases[i].vel.x += force_vec.x / world->particles[i].mass;
        world->phases[i].vel.y += force_vec.y / world->particles[i].mass;
    }
}

void *simulate(void *arg) {
    shared_data_t *shared = arg;
    world_t world;
    world.paused = false;
    init_world(&world);
    const uint64_t TARGET_TPS = 120;
    const world_time_t TICK = 1.0 / TARGET_TPS;
    uint64_t target_nsecs = BILLION / TARGET_TPS;
    struct timespec last_tick;
    clock_gettime(CLOCK_MONOTONIC, &last_tick);
    double lost_time = 0;
    pthread_t workers[NUM_WORKERS];
    step_helper_data_t data = {
        .world = &world,
        .dt = TICK,
        .max_force = 2,
        .next = NULL,
        .next_frame = NULL,
        .num_done = 0,
        .should_start_mask = 0,
        .main = PTHREAD_COND_INITIALIZER,
        .main_mutex = PTHREAD_MUTEX_INITIALIZER,
        .workers = PTHREAD_COND_INITIALIZER,
        .workers_mutex = PTHREAD_MUTEX_INITIALIZER,
    };
    for (size_t i = 0; i < NUM_WORKERS; i++) {
        step_helper_arg_t *arg = malloc(sizeof(step_helper_arg_t));
        arg->data = &data;
        arg->worker_id = i;
        pthread_create(&workers[i], NULL, step_helper, arg);
    }
    while (true) {
        input_t *input = atomic_exchange_explicit(&shared->input, NULL, __ATOMIC_RELAXED);
        frame_t *next_frame;
        if (world.paused) {
            process_input(input, &world, 0);
            next_frame = make_frame(&world);
        } else {
            world_time_t dt = TICK;// + lost_time;
            process_input(input, &world, dt);
            step(&data, dt, 2);
            next_frame = data.next_frame;
        }
        free(atomic_exchange_explicit(&shared->frame, next_frame, __ATOMIC_RELAXED));
        lost_time = (double) wait_tick(&last_tick, target_nsecs) / BILLION;
        (void) lost_time;
    }

    return NULL;
}


static frame_t *step(step_helper_data_t *data, world_time_t dt, float max_force) {
    pthread_mutex_lock(&data->main_mutex);

    // initialize next phase
    data->dt = dt;
    data->max_force = max_force;
    data->next = malloc(data->world->num_particles * sizeof(phase_pair_t));
    data->next_frame = malloc(sizeof(frame_t) + data->world->num_particles * sizeof(circle_t));
    data->next_frame->num_circles = data->world->num_particles;

    // wake up workers    
    data->should_start_mask = ((uint64_t) 1 << NUM_WORKERS) - 1;
    pthread_cond_broadcast(&data->workers);

    // wait for workers to finish
    while (data->num_done < NUM_WORKERS) {
        pthread_cond_wait(&data->main, &data->main_mutex);
    }
    data->num_done = 0;
    pthread_mutex_unlock(&data->main_mutex);

    // swap to next phase
    free(data->world->phases);
    data->world->phases = data->next;
    return data->next_frame;
}

static void *step_helper(void *arg) {
    step_helper_arg_t *helper_arg = arg;
    step_helper_data_t *data = helper_arg->data;
    world_t *world = data->world;
    size_t worker_id = helper_arg->worker_id;
    free(helper_arg);

    pthread_mutex_lock(&data->main_mutex);
    while (true) {
        while ((data->should_start_mask & ((uint64_t) 1 << worker_id)) == 0) {
            pthread_cond_wait(&data->workers, &data->main_mutex);
        }
        data->should_start_mask &= ~((uint64_t) 1 << worker_id);
        pthread_mutex_unlock(&data->main_mutex);

        world_time_t dt = data->dt;
        float max_force = data->max_force;
        phase_pair_t *next = data->next;
        size_t first_i = world->num_particles * worker_id / NUM_WORKERS;
        size_t last_i = world->num_particles * (worker_id + 1) / NUM_WORKERS;
        for (size_t i = first_i; i < last_i; i++) {
            particle_t *pai = &world->particles[i];
            phase_pair_t *pi = &next[i];
            pi->pos.x = world->phases[i].pos.x;
            pi->pos.y = world->phases[i].pos.y;
            pi->vel.x = world->phases[i].vel.x;
            pi->vel.y = world->phases[i].vel.y;
            for (size_t j = 0; j < world->num_particles; j++) {
                particle_t *paj = &world->particles[j];
                phase_pair_t *pj = &world->phases[j];
                dist_t dx = pi->pos.x - pj->pos.x;
                dist_t dy = pi->pos.y - pj->pos.y;
                float dist_sq = dx * dx + dy * dy;
                dist_sq = fmaxf(dist_sq, 0.1);
                dist_t dist = sqrtf(dist_sq);

                // total repulsion between i and j
                float force = 0;

                #ifdef GLOBAL_CLUSTERING
                force -= world->global_clustering * dist * sqrt(pi->mass * pj->mass);
                #endif

                // gravity
                #ifdef GRAVITY
                force -= world->gravity * pai->mass * paj->mass / dist_sq;
                #endif

                // charge
                #ifdef COULOMB
                force += world->coulomb * pi->charge * pj->charge / dist_sq;
                #endif

                // collision
                #ifdef COLLISION
                dist_t surface_dist = fmaxf(3 + dist - pai->radius - paj->radius, 0.1);
                if (surface_dist < 3) {
                    force += world->collision * pai->mass / powf(surface_dist, 3);
                }
                #endif

                // clustering
                #ifdef CLUSTERING
                dist_t target_cluster_dist = 5 + 8 * (pi->radius + pj->radius);
                dist_t dist_off = target_cluster_dist - dist;
                float cluster_func = dist_off / (target_cluster_dist * (1 + powf(dist_off/target_cluster_dist, 4)));
                force += world->clustering * pi->clustering * pj->clustering * cluster_func;
                #endif


                // clamp force
                force = fminf(force, max_force);

                // scale force by time
                force *= dt;

                // apply force
                vec_t force_vec = {
                    .x = force * dx / dist,
                    .y = force * dy / dist
                };
                pi->vel.x += force_vec.x / pai->mass;
                pi->vel.y += force_vec.y / pai->mass;
            }
        }

        // apply velocity
        for (size_t i = first_i; i < last_i; i++) {
            phase_pair_t *pi = &next[i];
            pi->pos.x += pi->vel.x * dt * !world->particles[i].fixed;
            pi->pos.y += pi->vel.y * dt * !world->particles[i].fixed;
        }

        // init_frame(data->next_frame, world, first_i, last_i);

        pthread_mutex_lock(&data->main_mutex);
        data->num_done++;
        pthread_cond_signal(&data->main);
    }
    pthread_mutex_unlock(&data->main_mutex);

    return NULL;
}