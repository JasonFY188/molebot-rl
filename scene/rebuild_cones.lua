-- rebuild_cones.lua
-- Rebuilds half_cone_R and half_cone_L from a FEW cubes instead of ~1,500 each,
-- so their collision is ~100x cheaper while they still drill the grains.
--
-- HOW TO RUN (simulation must be STOPPED):
--   Paste this whole block into CoppeliaSim's Lua console (the "Input code here" box
--   at the bottom) and press Enter. Then File > Save Scene (overwrite molebot.ttt).
--   Tunables: 'cube' = cube size (bigger = fewer cubes = faster/blockier).
--   If a cone ends up mirrored (flat face wrong way), flip the last true/false argument.

function rebuildHalfCone(name, baseR, height, cube, keepPosY)
    local ok, old = pcall(sim.getObject, '/' .. name)
    if (not ok) or old == nil or old < 0 then print('NOT FOUND: ' .. name); return end
    local parent = sim.getObjectParent(old)
    local layers = math.max(3, math.floor(height / cube + 0.5))
    local lh = height / layers
    local cubes = {}
    for li = 0, layers - 1 do
        local z = (li + 0.5) * lh
        local r = baseR * (1 - z / height)
        if r < cube then r = cube end
        local n = math.ceil(r / cube)
        for ix = -n, n do
            for iy = -n, n do
                local x = ix * cube
                local y = iy * cube
                local inHalf
                if keepPosY then inHalf = (y >= -0.001) else inHalf = (y <= 0.001) end
                if (x * x + y * y <= r * r) and inHalf then
                    local c = sim.createPrimitiveShape(sim.primitiveshape_cuboid, {cube, cube, lh})
                    sim.setObjectOrientation(c, {0, 0, 0}, old)   -- align to old cone's frame
                    sim.setObjectPosition(c, {x, y, z}, old)      -- local coords in that frame
                    cubes[#cubes + 1] = c
                end
            end
        end
    end
    local cone = sim.groupShapes(cubes, true)                     -- merge into ONE shape
    sim.setObjectInt32Param(cone, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(cone, sim.shapeintparam_respondable, 1)
    sim.computeMassAndInertia(cone, 700)
    sim.setObjectParent(cone, parent, true)                       -- reattach to the cone joint
    sim.removeObjects({old})
    sim.setObjectAlias(cone, name)
    print('rebuilt ' .. name .. ' with ' .. #cubes .. ' cubes')
end

-- dims measured from your model (baseRadius, height, cubeSize, keepPositiveY)
rebuildHalfCone('half_cone_R', 0.115, 0.19, 0.03, true)
rebuildHalfCone('half_cone_L', 0.125, 0.21, 0.03, false)
print('Done. Now File > Save Scene.')
