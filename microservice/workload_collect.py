"""Fresh validation/test only; training source remains the frozen 128 incidents."""
from microservice import regime_collect
regime_collect.SEEDS={**regime_collect.SEEDS,'validation':8802,'test':8803}
collect=regime_collect.collect
