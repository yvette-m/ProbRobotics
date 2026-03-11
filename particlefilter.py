# Yvette Roos
# CS141 Probabilistic Robotics
# 3/10/26

import argparse
import numpy as np
import cv2

# GLOBALS

NUM_PARTICLES       = 1000
MAX_ITERATIONS      = 1000
DRONE_VISION_RADIUS = 30    
PIXELS_PER_UNIT     = 50    # 1 map unit = 50 px (per spec)
MOVE_SPEED          = 1.0   # map units per step
MOVE_NOISE_STD      = 0.1  
PARTICLE_NOISE_STD  = 0.2   # noise added to particles
CONVERGENCE_RADIUS  = 1.5   # particles must be within this radius
CONVERGENCE_FRAC    = 0.90  # 90% must converge to finish simulation

# send random scouts every step to prevent false lock
INJECT_COUNT        = 50    # random particles per iteration
INJECT_NOISE_STD    = 0.1   # noise


# map set up

def map_bounds(img):
    # get the height and width in map units as per the spec
    h, w = img.shape[:2]
    return w / PIXELS_PER_UNIT / 2, h / PIXELS_PER_UNIT / 2

def px_to_map(px, img):
    # move origin to center
    h, w = img.shape[:2]
    return (np.asarray(px, dtype=float) - np.array([w / 2, h / 2])) / PIXELS_PER_UNIT

def map_to_px(mu, img):
    h, w = img.shape[:2]
    return np.asarray(mu, dtype=float) * PIXELS_PER_UNIT + np.array([w / 2, h / 2])


# helpers

def random_position(img):
    # random position for drone start
    h, w = img.shape[:2]
    r = DRONE_VISION_RADIUS
    return np.array([np.random.uniform(r, w - r), np.random.uniform(r, h - r)])


def init_particles(n, img):
    # random and uniform start for particles
    h, w = img.shape[:2]
    r = DRONE_VISION_RADIUS
    xs = np.random.uniform(r, w - r, n)
    ys = np.random.uniform(r, h - r, n)
    return np.column_stack([xs, ys])


def clamp_px(pts, img):
    # out of bounds check
    h, w = img.shape[:2]
    r = DRONE_VISION_RADIUS
    pts[:, 0] = np.clip(pts[:, 0], r, w - r)
    pts[:, 1] = np.clip(pts[:, 1], r, h - r)
    return pts


def move_drone(pos_px, img):

    # sample a random unit vector [dx, dy] with dx^2+dy^2=1.0 as per spec
    # apply move + N(0, sigma^2_movement) noise for the true new position.

    h, w = img.shape[:2]
    r = DRONE_VISION_RADIUS
    step_px = MOVE_SPEED * PIXELS_PER_UNIT

    for _ in range(10000):
        angle = np.random.uniform(0, 2 * np.pi)
        dx, dy = np.cos(angle), np.sin(angle)          

        nx = pos_px[0] + step_px * dx
        ny = pos_px[1] + step_px * dy

        if r <= nx < w - r and r <= ny < h - r:
            noise_px = MOVE_NOISE_STD * PIXELS_PER_UNIT * np.random.randn(2)
            new_pos = np.clip([nx + noise_px[0], ny + noise_px[1]],
                              [r, r], [w - r, h - r])
            return np.array(new_pos, dtype=float), np.array([dx, dy])

    return pos_px.copy(), np.array([0.0, 0.0])


# particle filter steps

def extract_patch(img, cx, cy, r):
    # get the patch around a point
    x0, y0 = int(cx) - r, int(cy) - r
    x1, y1 = x0 + 2 * r, y0 + 2 * r
    h, w = img.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return img[y0:y1, x0:x1]


def compute_weights(particles, drone_pos, img):
    # for each particle grap the map patch at the location and compare to drone observation
    r = DRONE_VISION_RADIUS
    drone_patch = extract_patch(img, drone_pos[0], drone_pos[1], r)
    if drone_patch is None:
        return np.ones(len(particles)) / len(particles)

    drone_gray = cv2.cvtColor(drone_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    weights = np.zeros(len(particles))

    for i, p in enumerate(particles):
        patch = extract_patch(img, p[0], p[1], r)
        if patch is None or patch.shape != drone_patch.shape:
            continue
        p_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        result = cv2.matchTemplate(p_gray, drone_gray, cv2.TM_CCOEFF_NORMED)
        weights[i] = max(0.0, float(result.max()))

    total = weights.sum()
    if total == 0:
        return np.ones(len(particles)) / len(particles)
    return weights / total


def resample(particles, weights, img):
    # weighted resampling, higher weight = more dups
    keep = len(particles) - INJECT_COUNT
    indices = np.random.choice(len(particles), size=keep, p=weights)
    noise_px = INJECT_NOISE_STD * PIXELS_PER_UNIT * np.random.randn(keep, 2)
    resampled = particles[indices] + noise_px

    h, w = img.shape[:2]
    r = DRONE_VISION_RADIUS
    scouts = np.column_stack([ # exploration mission in case particles are clustering wrong
        np.random.uniform(r, w - r, INJECT_COUNT),
        np.random.uniform(r, h - r, INJECT_COUNT),
    ])

    return clamp_px(np.vstack([resampled, scouts]), img)


def move_particles(particles, unit_vec, img):
    # moves the particles by the same amount the drone moves
    step_px = MOVE_SPEED * PIXELS_PER_UNIT
    noise_px = PARTICLE_NOISE_STD * PIXELS_PER_UNIT * np.random.randn(*particles.shape)
    return clamp_px(particles + step_px * unit_vec + noise_px, img)


def check_convergence(particles, drone_pos):
    #ignores the extra scouts to check convergence
    weighted = particles[:-INJECT_COUNT]
    dists_units = np.linalg.norm(weighted - drone_pos, axis=1) / PIXELS_PER_UNIT
    return np.mean(dists_units < CONVERGENCE_RADIUS) >= CONVERGENCE_FRAC


# visuals
def render(img, particles, drone_pos, weights, iteration):
    frame = img.copy()
    h, w = img.shape[:2]
    keep = len(particles) - INJECT_COUNT

    # weighted particles (purple)
    max_w = weights[:keep].max() if weights[:keep].max() > 0 else 1.0
    for i, p in enumerate(particles[:keep]):
        radius = max(1, int(1 + 6 * weights[i] / max_w))
        cv2.circle(frame, (int(p[0]), int(p[1])), radius, (128, 0, 128), -1)

    # scout particles (green)
    for p in particles[keep:]:
        cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 200, 0), -1)

    # actual drone position (yellow)
    dp = (int(drone_pos[0]), int(drone_pos[1]))
    cv2.circle(frame, dp, 20, (0, 255, 255), 2)
    cv2.circle(frame, dp, DRONE_VISION_RADIUS, (0, 255, 255), 1)

    # mean of weighted particles (red)
    mean_pos = particles[:keep].mean(axis=0)
    cv2.drawMarker(frame, (int(mean_pos[0]), int(mean_pos[1])),
                   (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.putText(frame, '+', (w // 2 - 8, h // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    drone_mu = px_to_map(drone_pos, img)
    mean_mu  = px_to_map(mean_pos,  img)
    cv2.putText(frame,
                f'Iter {iteration}  '
                f'drone:({drone_mu[0]:.2f},{drone_mu[1]:.2f})  '
                f'mean:({mean_mu[0]:.2f},{mean_mu[1]:.2f})',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow('Particle Filter', frame)


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map',        default='BayMap.png')
    parser.add_argument('--particles',  type=int, default=NUM_PARTICLES)
    parser.add_argument('--iters',      type=int, default=MAX_ITERATIONS)
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()

    img = cv2.imread(args.map)
    if img is None:
        raise FileNotFoundError(f"Could not load map: {args.map}")

    h, w = img.shape[:2]
    x_half, y_half = map_bounds(img)
    print(f"Map      : {w}x{h} px  →  "
          f"x∈[{-x_half:.1f}, {x_half:.1f}]  y∈[{-y_half:.1f}, {y_half:.1f}] map units")
    print(f"Particles: {args.particles}  (scouts: {INJECT_COUNT} per iter)")

    np.random.seed()
    drone_pos = random_position(img)
    particles = init_particles(args.particles, img)

    drone_mu = px_to_map(drone_pos, img)
    print(f"Drone start: px({drone_pos[0]:.0f},{drone_pos[1]:.0f})  "
          f"map({drone_mu[0]:.2f},{drone_mu[1]:.2f})")

    for iteration in range(1, args.iters + 1):

        # 1. Sense — compute P(z|x) weights via NCC
        weights = compute_weights(particles, drone_pos, img)

        # 2. Render
        if not args.no_display:
            render(img, particles, drone_pos, weights, iteration)
            if cv2.waitKey(0) & 0xFF == 27: 
                break

        # 3. Convergence check (weighted particles only)
        if check_convergence(particles, drone_pos):
            print(f"\nConverged after {iteration} iterations!")
            break

        # 4. Resample with injection
        particles = resample(particles, weights, img)

        # 5. Move drone — true pos gets hidden noise
        drone_pos, unit_vec = move_drone(drone_pos, img)

        # 6. Move particles — same unit vec, independent noise
        particles = move_particles(particles, unit_vec, img)

    # output
    mean_px  = particles[:-INJECT_COUNT].mean(axis=0)
    drone_mu = px_to_map(drone_pos, img)
    mean_mu  = px_to_map(mean_px,   img)
    err      = np.abs(drone_mu - mean_mu)

    print(f"\nFinal drone : ({drone_mu[0]:.3f}, {drone_mu[1]:.3f})")
    print(f"Final mean   : ({mean_mu[0]:.3f}, {mean_mu[1]:.3f})")
    print(f"Error        : ({err[0]:.3f}, {err[1]:.3f})")
    print(f"Total iterations        : {iteration}")

    if not args.no_display:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
