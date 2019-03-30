from yank.multistate import SAMSSampler, MultiStateReporter, ReplicaExchangeAnalyzer

iteration=1000
reporter = MultiStateReporter('traj.nc', open_mode='r',checkpoint_interval=1)
#sampler_states = list()
#for i in range(iteration):
#    sampler_states.append(reporter.read_sampler_states(iteration=i))
analyzer = ReplicaExchangeAnalyzer(reporter)
#Deltaf_ij, dDeltaf_ij = analyzer.get_free_energy()
print(analyzer.get_free_energy())
