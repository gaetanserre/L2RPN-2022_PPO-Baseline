from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from grid2op.Chronics.multiFolder import Multifolder

class GymEnvWithRecoWithDNWithShuffle(GymEnvWithRecoWithDN):    
    """this env shuffles the chronics order from time to time"""
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, **kwargs):
        super().__init__(env_init, *args, reward_cumul=reward_cumul, safe_max_rho=safe_max_rho, **kwargs)
        self.nb_reset = 0
        
    def reset(self, seed=None, return_info=False, options=None):
        # shuffle the chronics from time to time (to change the order in which they are 
        # seen by the agent)
        self.nb_reset += 1
        if isinstance(self.init_env.chronics_handler.real_data, Multifolder):
            nb_chron = len(self.init_env.chronics_handler.real_data._order)
            if self.nb_reset % nb_chron == 0:
                self.init_env.chronics_handler.reset()
        return super().reset(seed, return_info, options)