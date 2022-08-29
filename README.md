# mf6adj

## example input file

begin options

end options




begin performance_measure pm1 type residual
KPER KSTP K I J OBS WEIGHT
(repeat for multiple observations)
end performance_measure pm1

begin performance_measure pm2 type direct
KPER KSTP K I J WEIGHT (optional) # (todo extend to support multiple times)
 (repeat for multiple nodes, multiple times)
end performance_measure pm2

begin performance_measure pm3 type zone
KPER KSTP K I J #for a given time (todo extend to support multiple times)
(repeat for multiple nodes, multiple times)
end performance_measure pm3
