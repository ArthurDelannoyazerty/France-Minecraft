from litemapy import Schematic, Region, BlockState

# Shortcut to create a schematic with a single region
reg = Region(0, 0, 0, 21, 21, 21)
schem = reg.as_schematic(name="Planet", author="SmylerMC", description="Made with litemapy")

# Create the block state we are going to use
block = BlockState("minecraft:light_blue_concrete")

# Build the planet
for x, y, z in reg.block_positions():
    if round(((x-10)**2 + (y-10)**2 + (z-10)**2)**.5) <= 10:
        reg[x, y, z] = block

# Save the schematic
schem.save("data/litemapy/planet.litematic")