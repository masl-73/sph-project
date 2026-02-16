import numpy as np

def leapfrog_kick(velocities, accelerations, dt):
    """
    v(t + dt/2) = v(t) + a(t) * dt/2
    """
    return velocities + accelerations * (dt / 2.0)

def leapfrog_drift(positions, velocities_half, dt):
    """
    x(t + dt) = x(t) + v(t + dt/2) * dt
    """
    return positions + velocities_half * dt

def leapfrog_full_step(positions, velocities, accelerations, physics_func, dt, *args):
    """
    Standard Leap-frog step
    1. v(t+dt/2) = v(t) + a(t)*dt/2
    2. r(t+dt) = r(t) + v(t+dt/2)*dt
    3. a(t+dt) = physics(r(t+dt))
    4. v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
    """
    v_half = leapfrog_kick(velocities, accelerations, dt)
    positions_new = leapfrog_drift(positions, v_half, dt)
    
    # Compute new accelerations at positions_new
    accelerations_new = physics_func(positions_new, v_half, *args)
    
    velocities_new = leapfrog_kick(v_half, accelerations_new, dt)
    
    return positions_new, velocities_new, accelerations_new
