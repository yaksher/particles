#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <time.h>

#include "simulation.h"

#define BILLION 1000000000L

typedef double world_time_t;

typedef struct {
    point_t center;
    vec_t velocity;
    float mass;
    float charge;
    float clustering;
    float radius;
} particle_t;

typedef struct {
    size_t num_particles;
    particle_t *particles;
    bool paused;
    double gravity;
    double coulomb;
    double collision;
    double clustering;
    double global_clustering;
} world_t;

double clamp(double x, double min, double max) {
    if (x < min) {
        return min;
    } else if (x > max) {
        return max;
    } else {
        return x;
    }
}

double unif_rand() {
    return rand() / (double) RAND_MAX;
}

double norm_rand() {
    const double EPSILON = 1e-12;
    double u;
    do {
        u = unif_rand();
    } while (u <= EPSILON);
    double v = unif_rand();
    return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

float radius_of_mass(float mass) {
    return sqrt(mass) / 2;
}

#define CHARGE_RANGE 5
float particle_shade(particle_t *particle) {
    return (particle->charge + CHARGE_RANGE) / (2 * CHARGE_RANGE);
}
// compute the frame from the world state
frame_t *make_frame(world_t *world) {
    frame_t *frame = malloc(sizeof(frame_t) + world->num_particles * sizeof(circle_t));
    frame->num_circles = world->num_particles;
    circle_t *circles = frame->circles;
    for (size_t i = 0; i < frame->num_circles; i++) {
        circles[i].center.x = world->particles[i].center.x;
        circles[i].center.y = world->particles[i].center.y;
        circles[i].radius = world->particles[i].radius * 1.2;
        circles[i].shade = particle_shade(&world->particles[i]);
    }
    return frame;
}

// wait until the next tick
// if the last tick took too long, return the time that needs to be
// made up
uint64_t wait_tick(struct timespec *last_tick, uint64_t target_nsecs) {
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
// #define COLLISION
#define CLUSTERING
// #define GLOBAL_CLUSTERING
void init_world(world_t *world) {
    const size_t NUM_PARTICLES = 1000;
    *world = (world_t) {
        .paused = false,
        .gravity = 180,
        .coulomb = 0,
        .collision = 0,
        .clustering = 6,
        .global_clustering = 1. / NUM_PARTICLES,
        .num_particles = NUM_PARTICLES,
        .particles = malloc(NUM_PARTICLES * sizeof(particle_t))
    };
    world->particles[0].center.x = 0;
    world->particles[0].center.y = 0;
    world->particles[0].velocity.x = 0;
    world->particles[0].velocity.y = 0;
    world->particles[0].mass = 100000;
    world->particles[0].charge = 0;
    world->particles[0].clustering = 0;
    world->particles[0].radius = 50;
    const float RING_RADIUS = 800;
    const float VELOCITY = 4;
    for (size_t i = 1; i < world->num_particles; i++) {
        particle_t particle;
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
        float rad = RING_RADIUS / inv_rad;
        // add some noise
        rad *= norm_rand() * 0.01 + 1;
        // scale the direction vector to the desired radius
        particle.center.x = dir.x * rad;
        particle.center.y = dir.y * rad;

        // Init velocity (approx. tangential to the ring):
        particle.velocity.x = particle.center.y / sqrtf(rad) * VELOCITY + norm_rand() * 0.1;
        particle.velocity.y = -particle.center.x / sqrtf(rad) * VELOCITY + norm_rand() * 0.1;

        // Init mass, charge, clustering, and radius:
        // particle.mass = fmax(norm_rand() * 5 + 20, 0.01);
        // particle.radius = radius_of_mass(particle.mass);
        particle.mass = 1;
        particle.radius = 3;
        // particle.charge = clamp(norm_rand() * 5, -CHARGE_RANGE, CHARGE_RANGE);
        // particle.charge = unif_rand() * 2 * CHARGE_RANGE - CHARGE_RANGE;
        particle.charge = unif_rand() < 0.5 ? CHARGE_RANGE : -CHARGE_RANGE;
        particle.clustering = 1;
        world->particles[i] = particle;
    }
}

void step(world_t *, world_time_t, float);

void process_input(input_t *input, world_t *world, world_time_t dt) {
    if (input == NULL) {
        return;
    }
    switch (input->type) {
        case INPUT_TYPE_FORCE_POINT: {
            input_force_point_t *force_point = &input->force_point;
            for (size_t i = 0; i < world->num_particles; i++) {
                float dx = world->particles[i].center.x - force_point->point.x;
                float dy = world->particles[i].center.y - force_point->point.y;
                float dist_sq = dx * dx + dy * dy;
                float dist = sqrtf(dist_sq);
                float force = 0;
                switch (force_point->force_type) {
                    case FORCE_TYPE_GRAVITY:
                        force = world->gravity * world->particles[i].mass * force_point->strength / dist_sq;
                        break;
                    case FORCE_TYPE_COULOMB:
                        force = world->coulomb * world->particles[i].charge * force_point->strength / dist_sq;
                        break;
                }
                force *= dt;
                vec_t force_vec = {
                    .x = force * dx / dist,
                    .y = force * dy / dist
                };
                world->particles[i].velocity.x += force_vec.x / world->particles[i].mass;
                world->particles[i].velocity.y += force_vec.y / world->particles[i].mass;
            }
            break;
        }
        case INPUT_TYPE_SPAWN: {
            input_spawn_t *spawn = &input->spawn;
            world->particles = realloc(world->particles, (world->num_particles + 1) * sizeof(particle_t));
            particle_t particle;
            particle.center.x = spawn->point.x;
            particle.center.y = spawn->point.y;
            particle.velocity.x = 0;
            particle.velocity.y = 0;
            particle.mass = spawn->mass;
            particle.charge = spawn->charge;
            particle.clustering = 1;
            particle.radius = radius_of_mass(particle.mass);
            world->particles[world->num_particles] = particle;
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
                    init_world(world);
                    break;
            }
            break;
        }
    }
}

void *simulate(void *arg) {
    shared_data_t *shared = arg;
    world_t world;
    init_world(&world);
    const uint64_t TARGET_TPS = 120;
    const world_time_t TICK = 1.0 / TARGET_TPS;
    uint64_t target_nsecs = BILLION / TARGET_TPS;
    struct timespec last_tick;
    clock_gettime(CLOCK_MONOTONIC, &last_tick);
    double lost_time = 0;
    while (true) {
        input_t *input = atomic_exchange_explicit(&shared->input, NULL, __ATOMIC_RELAXED);
        world_time_t dt = world.paused ? 0 : TICK + lost_time;
        process_input(input, &world, dt);
        step(&world, dt, 5);
        free(atomic_exchange_explicit(&shared->frame, make_frame(&world), __ATOMIC_RELAXED));
        lost_time = (double) wait_tick(&last_tick, target_nsecs) / BILLION;
    }

    return NULL;
}


void step(world_t *world, world_time_t dt, float max_force) {
    for (size_t i = 0; i < world->num_particles; i++) {
        for (size_t j = i + 1; j < world->num_particles; j++) {
            particle_t *pi = &world->particles[i], *pj = &world->particles[j];
            float dx = pi->center.x - pj->center.x;
            float dy = pi->center.y - pj->center.y;
            float dist_sq = dx * dx + dy * dy;
            dist_sq = fmaxf(dist_sq, 0.1);
            float dist = sqrtf(dist_sq);

            // total repulsion between i and j
            float force = 0;

            #ifdef GLOBAL_CLUSTERING
            force -= world->global_clustering * dist * sqrt(pi->mass * pj->mass);
            #endif

            // gravity
            #ifdef GRAVITY
            force -= world->gravity * pi->mass * pj->mass / dist_sq;
            #endif

            // charge
            #ifdef COULOMB
            force += world->coulomb * pi->charge * pj->charge / dist_sq;
            #endif

            // collision
            #ifdef COLLISION
            float surface_dist = fmaxf(3 + dist - pi->radius - pj->radius, 0.1);
            if (surface_dist < 3) {
                force += world->collision / powf(surface_dist, 3);
            }
            #endif

            // clustering
            #ifdef CLUSTERING
            float target_cluster_dist = 4 + 4 * (pi->radius + pj->radius);
            float dist_off = target_cluster_dist - dist;
            float cluster_func = dist_off / (target_cluster_dist * (1 + powf(dist_off/target_cluster_dist, 6)));
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
            world->particles[i].velocity.x += force_vec.x / world->particles[i].mass;
            world->particles[i].velocity.y += force_vec.y / world->particles[i].mass;
            world->particles[j].velocity.x -= force_vec.x / world->particles[j].mass;
            world->particles[j].velocity.y -= force_vec.y / world->particles[j].mass;
        }
    }

    // apply velocity
    for (size_t i = 0; i < world->num_particles; i++) {
        // float vel_dot = world->particles[i].velocity.x * world->particles[i].center.x
        //               + world->particles[i].velocity.y * world->particles[i].center.y;
        // float max_vel = max_step / dt;
        // if (vel_dot > max_vel * max_vel) {
        //     float vel = sqrt(vel_dot);
        //     world->particles[i].velocity.x *= max_vel / vel;
        //     world->particles[i].velocity.y *= max_vel / vel;
        // }
        world->particles[i].center.x += world->particles[i].velocity.x * dt;
        world->particles[i].center.y += world->particles[i].velocity.y * dt;
    }
}
