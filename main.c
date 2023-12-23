#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>
#include <math.h>
#include <assert.h>

#include <SDL.h>

#include "communication.h"
#include "simulation.h"

// Define MAX and MIN macros
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

// Define screen dimensions
#define DEFAULT_SCREEN_WIDTH 1600
#define DEFAULT_SCREEN_HEIGHT 1200

typedef int32_t pt_t;

typedef struct {
    // number of pixels per world unit
    double scale;
    // frame time in milliseconds
    uint64_t frame_time;
    // center of the view in world coordinates
    point_t center;
    // window dimensions in pixels
    pt_t window_width;
    pt_t window_height;
    double pt_to_pixel;
} view_t;

float len_screen_to_world(view_t *state, pt_t len) {
    return len / state->scale;
}

int32_t len_world_to_screen(view_t *state, float dist) {
    return dist * state->scale * state->pt_to_pixel;
}

SDL_Point pos_world_to_screen(view_t *state, point_t pos) {
    return (SDL_Point) {
        .x = len_world_to_screen(state, 
            pos.x - state->center.x + len_screen_to_world(state, state->window_width / 2)
        ),
        .y = len_world_to_screen(state, 
            pos.y - state->center.y + len_screen_to_world(state, state->window_height / 2)
        )
    };
}

void add_input(shared_data_t *shared, input_t *input) {
    input->base.next = NULL;
    input_t *old_input = atomic_exchange_explicit(&shared->input, NULL, __ATOMIC_RELAXED);
    // this is a FILO queue which is suboptimal, but it's good enough for now
    input->base.next = old_input;
    input_t *intermediate = atomic_exchange_explicit(&shared->input, input, __ATOMIC_RELAXED);
    assert(intermediate == NULL && "only this thread should ever put non-NULL values here");
}

void add_command(shared_data_t *shared, int command_type) {
    input_t *input = malloc(sizeof(input_t));
    input->base.type = INPUT_TYPE_COMMAND;
    input->command.command_type = command_type;
    add_input(shared, (input_t *) input);
}

// returns true if should quit and false otherwise
bool process_events(view_t *state, shared_data_t *shared) {
    SDL_Event e;
    static bool dragging = false;
    static bool paused = false;

    // Handle all waiting events
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
            case SDL_WINDOWEVENT: {
                switch (e.window.event) {
                    case SDL_WINDOWEVENT_RESIZED: {
                        break;
                    }
                    case SDL_WINDOWEVENT_SIZE_CHANGED: {
                        state->window_width = e.window.data1;
                        state->window_height = e.window.data2;
                        break;
                    }
                    case SDL_WINDOWEVENT_MINIMIZED: {
                        break;
                    }
                    case SDL_WINDOWEVENT_MAXIMIZED: {
                        break;
                    }
                    case SDL_WINDOWEVENT_RESTORED: {
                        break;
                    }
                    case SDL_WINDOWEVENT_ENTER: {
                        break;
                    }
                    case SDL_WINDOWEVENT_LEAVE: {
                        break;
                    }
                    case SDL_WINDOWEVENT_FOCUS_GAINED: {
                        break;
                    }
                    case SDL_WINDOWEVENT_FOCUS_LOST: {
                        break;
                    }
                    case SDL_WINDOWEVENT_CLOSE: {
                        break;
                    }
                    case SDL_WINDOWEVENT_TAKE_FOCUS: {
                        break;
                    }
                    case SDL_WINDOWEVENT_HIT_TEST: {
                        break;
                    }
                }
                break;
            }
            case SDL_KEYDOWN: {
                break;
            }
            case SDL_KEYUP: {
                if (e.key.keysym.sym == SDLK_r) {
                    add_command(shared, COMMAND_TYPE_RESET);
                } else if (e.key.keysym.sym == SDLK_SPACE) {
                    add_command(shared, 
                        paused ? COMMAND_TYPE_RESUME : COMMAND_TYPE_PAUSE
                    );
                    paused = !paused;
                }
                break;
            }
            case SDL_MOUSEBUTTONDOWN: {
                if (e.button.button == SDL_BUTTON_RIGHT) {
                    dragging = true;
                }
                if (e.button.button == SDL_BUTTON_LEFT) {
                    SDL_Point pos = (SDL_Point) { e.button.x, e.button.y };
                    point_t world_pos = (point_t) {
                        .x = state->center.x + len_screen_to_world(state, pos.x - state->window_width / 2),
                        .y = state->center.y + len_screen_to_world(state, pos.y - state->window_height / 2)
                    };
                    input_t *input = malloc(sizeof(input_t));
                    input->base.type = INPUT_TYPE_SPAWN;
                    input->spawn.point = world_pos;
                    input->spawn.mass = 100;
                    input->spawn.charge = rand() / (float) RAND_MAX * 10 - 5;
                    add_input(shared, input);
                }
                break;
            }
            case SDL_MOUSEBUTTONUP: {
                if (e.button.button == SDL_BUTTON_RIGHT) {
                    dragging = false;
                }
                break;
            }
            case SDL_MOUSEMOTION: {
                if (dragging) {
                    state->center.x -= e.motion.xrel / state->scale;
                    state->center.y -= e.motion.yrel / state->scale;
                }
                break;
            }
            case SDL_MOUSEWHEEL: {
                if (e.wheel.y > 0) {
                    state->scale *= 1.1;
                } else if (e.wheel.y < 0) {
                    state->scale /= 1.1;
                }
                break;
            }
            case SDL_TEXTEDITING: {
                break;
            }
            case SDL_TEXTINPUT: {
                break;
            }
        }
        // User requests quit
        if (e.type == SDL_QUIT) {
            fprintf(stderr, "SDL_QUIT\n");
            return true;
        }
    }

    // User requests quit
    return false;
}

uint32_t round_up_to_mul_8(uint32_t v) {
    return (v + (8 - 1)) & ~(8 - 1);
}

// draws a circle with radius r at center, using the point buffer to store the points
// the point buffer must be at least 4r^2 + 1 points long
// TODO: Optimize to draw a square and then fill in the edges
void draw_circle(SDL_Renderer *renderer, SDL_Point center, pt_t radius, SDL_Point *point_buffer) {
    // array of points, upper bounded by 4r^2 + 1
    size_t num_points = 0;
    for (pt_t w = 0; w < radius * 2 + 1; w++) {
        for (pt_t h = 0; h < radius * 2 + 1; h++) {
            int dx = radius - w; // horizontal offset
            int dy = radius - h; // vertical offset
            if ((dx*dx + dy*dy) <= (radius * radius)) {
                point_buffer[num_points++] = (SDL_Point) { center.x + dx, center.y + dy };
            }
        }
    }
    SDL_RenderDrawPoints(renderer, point_buffer, num_points);
}

void draw_frame(SDL_Renderer *renderer, view_t *state, shared_data_t *shared, double mean_fps) {
    static frame_t *last_frame = NULL;
    // Initialize renderer color black for the background
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);

    // Clear screen
    SDL_RenderClear(renderer);

    // aligned access is atomic; we'll see a coherent frame, regardless of what
    // the simulation thread is doing
    frame_t *frame = atomic_exchange_explicit(&shared->frame, NULL, __ATOMIC_RELAXED);
    if (!frame) {
        // if we didn't get a frame, just use the last one
        frame = last_frame;
    } else {
        // if we did get a frame, free the last one
        free(last_frame);
        last_frame = frame;
    }
    if (!frame) {
        // if we don't have a frame at all, just return after clearing the canvas.
        return;
    }


    size_t num_circles = frame->num_circles;
    circle_t *circles = frame->circles;
    // Draw filled square
    // size_t points_array_size = 0;
    // for (size_t i = 0; i < num_circles; i++) {
    //     points_array_size += round_up_to_mul_8(
    //         len_world_to_screen(state, circles[i].radius) * 8 * 35 / 49
    //     );
    // }
    // SDL_Rect *rects = malloc(num_circles * sizeof(SDL_Rect));
    size_t point_buf_size = 0;
    SDL_Point *circle_point_buffer = NULL;
    for (size_t i = 0; i < num_circles; i++) {
        SDL_Point center = pos_world_to_screen(state, circles[i].center);
        pt_t radius = len_world_to_screen(state, circles[i].radius);
        if (radius * radius * 4 + 1 > point_buf_size) {
            point_buf_size = radius * radius * 4 + 1;
            free(circle_point_buffer);
            circle_point_buffer = malloc(point_buf_size * sizeof(SDL_Point));
        }
        SDL_SetRenderDrawColor(renderer, 
            0xFF * circles[i].shade,
            0x80, 
            0xFF * (1 - circles[i].shade),
            0xFF
        );
        draw_circle(renderer, center, radius, circle_point_buffer);
    }
    free(circle_point_buffer);

    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
    int32_t off = 20 * state->pt_to_pixel;
    int32_t h = 100 * state->pt_to_pixel;
    int32_t w = 10 * state->pt_to_pixel;
    SDL_Rect rect = {
        .x = off,
        .y = off,
        .w = w,
        .h = h
    };
    SDL_RenderDrawRect(renderer, &rect);
    double norm_fps = mean_fps / 240;
    rect.h = h * norm_fps;
    rect.y = off + h - rect.h;
    SDL_RenderFillRect(renderer, &rect);
    

    // Update screen
    SDL_RenderPresent(renderer);
}

int main(int argc, char *argv[]) {
    // Unused argc, argv
    (void)argc;
    (void)argv;
    int ret_code = EXIT_SUCCESS;

    // start simulation
    pthread_t simulation_thread;
    shared_data_t shared = {
            .frame = NULL,
            .input = NULL
    };
    view_t state = {
            .scale = 0.5,
            .frame_time = 1000/60,
            .window_width = DEFAULT_SCREEN_WIDTH,
            .window_height = DEFAULT_SCREEN_HEIGHT
    };
    if (pthread_create(&simulation_thread, NULL, simulate, &shared)) {
        fprintf(stderr, "Failed to create simulation thread!\n");
        return EXIT_FAILURE;
    }

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL could not be initialized!\n"
               "SDL_Error: %s\n",
               SDL_GetError());
        return EXIT_FAILURE;
    }

#if defined linux && SDL_VERSION_ATLEAST(2, 0, 8)
    // Disable compositor bypass
    if (!SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "0")) {
        fprintf(stderr, "SDL can not disable compositor bypass!\n");
        return EXIT_FAILURE;
    }
#endif

    // Create window
    SDL_Window *window = SDL_CreateWindow("Particles",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT,
                                          SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
                                        | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window) {
        fprintf(stderr, "Window could not be created!\n"
               "SDL_Error: %s\n",
               SDL_GetError());
        ret_code = EXIT_FAILURE;
        goto CLEANUP_SDL;
    }
    // Create renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Renderer could not be created!\n"
                "SDL_Error: %s\n",
                SDL_GetError());
        ret_code = EXIT_FAILURE;
        goto CLEANUP_WINDOW;
    }

    int pwidth, pheight, lwidth, lheight;
    SDL_GetWindowSize(window, &pwidth, &pheight);
    state.window_width = pwidth;
    state.window_height = pheight;
    SDL_GetRendererOutputSize(renderer, &lwidth, &lheight);
    double wpt_to_pixel = (double) lwidth / state.window_width;
    double hpt_to_pixel = (double) lheight / state.window_height;
    state.pt_to_pixel = MIN(wpt_to_pixel, hpt_to_pixel);
    if (wpt_to_pixel != hpt_to_pixel) {
        fprintf(stderr, "Warning: wpt_to_pixel = %.6f != %.6f = hpt_to_pixel.\n", wpt_to_pixel, hpt_to_pixel);
    }

    uint64_t last_frame = SDL_GetTicks64();
    double mean_fps = 60;
    // Event loop
    while (true) {
        // Handle events on queue
        if (process_events(&state, &shared)) {
            break;
        }
        draw_frame(renderer, &state, &shared, mean_fps);
        uint64_t frame_time = SDL_GetTicks64() - last_frame;
        if (4 * frame_time < state.frame_time) {
            mean_fps = 0.9 * mean_fps + 0.1 * 1000.0 * 4 / state.frame_time;
        } else {
            mean_fps = 0.9 * mean_fps + 0.1 * 1000.0 / frame_time;
        }
        int64_t time_left = state.frame_time - frame_time;
        if (time_left > 0) {
            SDL_Delay(time_left);
        }
        last_frame = SDL_GetTicks64();
    }

    // Destroy renderer
    SDL_DestroyRenderer(renderer);

    CLEANUP_WINDOW:
    // Destroy window
    SDL_DestroyWindow(window);

    CLEANUP_SDL:
    // Quit SDL
    SDL_Quit();

    return ret_code;
}
