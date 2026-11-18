"""Convert pinned Dreamer vector decoder predictions to observation units."""
def observation_prediction(prediction,symlog,xp):
    # Inverse transform each particle before averaging, not the particle mean.
    return xp.sign(prediction)*xp.expm1(xp.abs(prediction)) if symlog else prediction
