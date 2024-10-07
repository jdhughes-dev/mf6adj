import mf6adj

adj = mf6adj.Mf6Adj("IE_perfmeas.dat", "libmf6_arm.dylib",verbose_level=1)
adj.solve_gwf()
adj.solve_adjoint()