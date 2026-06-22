param(
    [int]$Rocks = 18,
    [int]$Trials = 4,
    [int]$Clusters = 5
)

python -m moon_rock_stack.run_experiment --rocks $Rocks --trials $Trials --clusters $Clusters
