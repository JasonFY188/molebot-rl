-- grain_field.lua
-- Paste this as the child script attached to the "dummy" object (replacing the
-- original ball-generation script).  It is a drop-in superset of your script:
--   * grain grid size / diameter / centre are read from signals set by the RL env
--     (falls back to your original 8x8x4 defaults if the env isn't driving it),
--   * it publishes grain-field statistics every step (float signals) that the RL
--     observation + dig-through reward consume:
--       grain_com_x/y/z   : centre of mass of the pile
--       grain_mean_disp    : mean distance each grain has moved from its start
--       grain_disturbed    : fraction of grains displaced past grain_disturb_thresh
-- The inter-grain repulsion + damping physics below is unchanged from your script.

function createPile()
    grains = {}

    -- ===== read arena / grid params from the RL env (with your defaults) =====
    local nx = sim.getInt32Signal('grain_nx') or 8
    local ny = sim.getInt32Signal('grain_ny') or 8
    local nz = sim.getInt32Signal('grain_nz') or 4
    local d  = sim.getFloatSignal('grain_d') or 0.1          -- ball diameter (m)
    local cx = sim.getFloatSignal('grain_cx') or 0.0         -- block centre X
    local cy = sim.getFloatSignal('grain_cy') or 0.0         -- block centre Y
    local floorZ = sim.getFloatSignal('grain_floorz') or 0.0 -- arena floor top
    disturbThresh = sim.getFloatSignal('grain_disturb_thresh') or 0.03
    -- spacing >= 2.0 keeps neighbours OUTSIDE the repulsion cutoff (rCut/d = 2.0) at
    -- spawn, so the pile does NOT explode (which otherwise stalls/crashes the solver).
    local spacing = sim.getFloatSignal('grain_spacing') or 2.1
    -- ========================================================================

    local s  = d * spacing
    local x0 = cx - (nx - 1) * s / 2
    local y0 = cy - (ny - 1) * s / 2
    local z0 = floorZ + d / 2 + 0.005

    for ix = 0, nx - 1 do
        for iy = 0, ny - 1 do
            for iz = 0, nz - 1 do
                local b = sim.createPrimitiveShape(sim.primitiveshape_spheroid, {d, d, d})
                sim.setObjectPosition(b, {x0 + ix * s, y0 + iy * s, z0 + iz * s}, -1)
                sim.setObjectInt32Param(b, sim.shapeintparam_static, 0)
                sim.setObjectInt32Param(b, sim.shapeintparam_respondable, 1)
                sim.computeMassAndInertia(b, 700)
                grains[#grains + 1] = b
            end
        end
    end

    -- remember each grain's initial position (for the "disturbed" metric). This gets
    -- overwritten once the pile has physically settled (see settleCountdown below) -
    -- balls spawn spaced apart and drop/roll into contact, and that natural settling
    -- alone can exceed disturbThresh, which would wrongly count as "robot disturbed it".
    initPos = {}
    for i = 1, #grains do
        initPos[i] = sim.getObjectPosition(grains[i], -1)
    end
    settleCountdown = 15   -- actuation steps to let the pile settle before it's "at rest"
end

function sysCall_init()
    createPile()
end

function sysCall_actuation()
    local n = #grains
    if n == 0 then return end
    -- reset the pile: delete every grain and regenerate the original block (reliable;
    -- teleporting dynamic objects mid-simulation does not sync with the physics engine)
    if sim.getInt32Signal('grain_reset') == 1 then
        for i = 1, n do pcall(sim.removeObject, grains[i]) end
        createPile()
        sim.setInt32Signal('grain_reset', 0)
        return
    end
    -- once the freshly (re)spawned pile has settled, re-baseline "disturbed" from its
    -- actual resting positions instead of the spawn positions
    if settleCountdown and settleCountdown > 0 then
        settleCountdown = settleCountdown - 1
        if settleCountdown == 0 then
            for i = 1, n do initPos[i] = sim.getObjectPosition(grains[i], -1) end
        end
    end
    -- Soft, capped, damped inter-grain repulsion (stable under compression).
    -- rCut  : interaction range;  kSoft : stiffness;  fMax : per-pair force cap (N);
    -- c     : velocity damping. The cap + overlap-proportional force can't explode.
    local rCut, kSoft, fMax, c = 0.20, 12.0, 1.5, 0.25
    local pos = {}
    for i = 1, n do pos[i] = sim.getObjectPosition(grains[i], -1) end
    local force = {}
    for i = 1, n do force[i] = {0, 0, 0} end
    local buckets, ci = {}, {}
    for i = 1, n do
        local p = pos[i]
        local cx = math.floor(p[1] / rCut)
        local cy = math.floor(p[2] / rCut)
        local cz = math.floor(p[3] / rCut)
        ci[i] = {cx, cy, cz}
        local key = cx .. "_" .. cy .. "_" .. cz
        local b = buckets[key]
        if not b then b = {}; buckets[key] = b end
        b[#b + 1] = i
    end
    for i = 1, n do
        local cc, pi = ci[i], pos[i]
        for ox = -1, 1 do for oy = -1, 1 do for oz = -1, 1 do
            local b = buckets[(cc[1] + ox) .. "_" .. (cc[2] + oy) .. "_" .. (cc[3] + oz)]
            if b then
                for bi = 1, #b do
                    local j = b[bi]
                    if j > i then
                        local dx, dy, dz = pos[j][1] - pi[1], pos[j][2] - pi[2], pos[j][3] - pi[3]
                        local dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                        if dist > 1e-4 and dist < rCut then
                            local mag = kSoft * (rCut - dist)   -- soft: grows with overlap
                            if mag > fMax then mag = fMax end   -- capped: cannot explode
                            local ux, uy, uz = dx / dist, dy / dist, dz / dist
                            force[i][1] = force[i][1] + mag * ux; force[i][2] = force[i][2] + mag * uy; force[i][3] = force[i][3] + mag * uz
                            force[j][1] = force[j][1] - mag * ux; force[j][2] = force[j][2] - mag * uy; force[j][3] = force[j][3] - mag * uz
                        end
                    end
                end
            end
        end end end
    end
    for i = 1, n do
        local lin = sim.getObjectVelocity(grains[i])
        sim.addForceAndTorque(grains[i],
            {force[i][1] - c * lin[1], force[i][2] - c * lin[2], force[i][3] - c * lin[3]}, {0, 0, 0})
    end

    -- ===== publish grain-field statistics for the RL agent =====
    local sx, sy, sz, meandisp, disturbed = 0, 0, 0, 0, 0
    for i = 1, n do
        local p = pos[i]
        sx = sx + p[1]; sy = sy + p[2]; sz = sz + p[3]
        local ip = initPos[i]
        local ex, ey, ez = p[1] - ip[1], p[2] - ip[2], p[3] - ip[3]
        local dd = math.sqrt(ex * ex + ey * ey + ez * ez)
        meandisp = meandisp + dd
        if dd > disturbThresh then disturbed = disturbed + 1 end
    end
    sim.setFloatSignal('grain_com_x', sx / n)
    sim.setFloatSignal('grain_com_y', sy / n)
    sim.setFloatSignal('grain_com_z', sz / n)
    sim.setFloatSignal('grain_mean_disp', meandisp / n)
    sim.setFloatSignal('grain_disturbed', disturbed / n)
end
