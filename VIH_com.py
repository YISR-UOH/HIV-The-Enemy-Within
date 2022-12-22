import numpy as np
# Model design
import agentpy as ap

# Visualization
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
import matplotlib.pyplot as plt
import seaborn as sns
import IPython


class Cell(ap.Agent):

    def setup(self):
        """ Initiate agent attributes. """
        self.grid = self.model.grid
        self.condition = np.random.default_rng().binomial(1, self.p.vih, size=1)[0]
        self.next_condition = self.condition
        self.dead_time = 0


        
class HIVmodel(ap.Model):

    def setup(self):

        # Create agents (trees)
        n_cells = int(self.p.size*self.p.size)
        # Create grid (system)
        self.grid = ap.Grid(self, (self.p.size, self.p.size), track_empty=True)
        self.agents = ap.AgentList(self, n_cells, Cell)
        self.grid.add_agents(self.agents, random=True, empty=True)

        # Initiate a dynamic variable for all trees
        # Condition 0: Alive, 1: Burning, 2: Burned 3: recovered
    def update(self):
        """ Record variables after setup and each step. """

        # Record share of agents with each condition
        for i, c in enumerate(('H', 'A1', 'A2', 'D')):
            n_agents = len(self.agents.select(self.agents.condition == i))
            self[c] = n_agents / int(self.p.size*self.p.size)
            self.record(c)
        self.agents.condition = self.agents.next_condition
        
        # Stop simulation if disease is gone
        #if self.H >= 1: self.stop()
        #if self.D >= 1: self.stop()

    def step(self):
        #segmentar en 4
        healty = self.agents.select(self.agents.condition == 0)
        infected_A1 = self.agents.select(self.agents.condition == 1)
        infected_A2 = self.agents.select(self.agents.condition == 2)
        dead_cells = self.agents.select(self.agents.condition == 3)
        if self.p.therapy == 1 and self.t >= 300:
            
            if self.p.lineal == 1:
                respond = (1 - (1/self.p.steps)*(self.t))
            else :
                respond = (self.p.respond-0.10)/2
            for h in healty:
                neighbors = self.grid.neighbors(h)
                numberOfA1 = 0
                numberOfA2 = 0
                h.next_condition = h.condition 
                for n in neighbors:
                    if n.condition == 2:
                        numberOfA2 += 1
                        if numberOfA2 >= 3:
                            h.next_condition = 1
                            break
                    if n.condition == 1:
                        numberOfA1 += 1
                        if numberOfA2 >= self.p.rankLevel:
                            if np.random.default_rng().binomial(1, ((1-respond)*(self.p.rankLevel/2)), size=1)[0]:
                                h.next_condition = 1
                            break
        
        else:
          for h in healty:
            neighbors = self.grid.neighbors(h)
            numberOfA2 = 0
            h.next_condition = h.condition
            for n in neighbors:
                if n.condition == 2:
                    numberOfA2 += 1
                    if numberOfA2 >= 3:
                        h.next_condition = 1
                        break
                if n.condition == 1:
                    h.next_condition = 1
                    break
            
        infected_A1.next_condition = 2
        infected_A2.next_condition = 3
        for d in dead_cells:
            if d.dead_time >= 4:
                if np.random.default_rng().binomial(1, self.p.replace, size=1)[0]:
                    if np.random.default_rng().binomial(1, self.p.infected, size=1)[0]:
                        d.next_condition = 1
                        d.dead_time = 0
                    else:
                        d.next_condition = 0
                        d.dead_time = 0
            else:
                d.next_condition = 3
                d.dead_time += 1

def virus_stackplot(data, ax):
    """ Stackplot of people's condition over time. """
    x = data.index.get_level_values('t')
    y = [data[var] for var in ['H', 'A1', 'A2', 'D']]
    
    color_dict = ['#00fc0a', '#f2fc00', '#fc8800','#000003']
    sns.set()
    ax.stackplot(x, y, labels=['healty', 'inf_A1', 'inf_A2', 'dead'],
                 colors =color_dict)

    ax.legend()
    ax.set_xlim(0, max(1, len(x)-1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Percentage of cells")
    
def animation_plot(model, axs):
    ax1,ax2 = axs
    
    ax1.set_title(f"Simulation of a HIV\n")
    virus_stackplot(model.output.variables.HIVmodel, ax1)
    
    attr_grid = model.grid.attr_grid('condition')
    color_dict = {0:'#00fc0a', 1:'#f2fc00', 2:'#fc8800',3:'#000003'}
    ap.gridplot(attr_grid, ax=ax2, color_dict=color_dict, convert=True)
    ax2.set_title(f"Simulation of a HIV\n"
                 f"Time-step: {model.t}, Total cell: "
                 f"{len(model.agents)}\n"
                 f"Healty: "
                 f"{len(model.agents.select(model.agents.condition == 0))}"
                 f"  dead: "
                 f"{len(model.agents.select(model.agents.condition == 3))}\n"
                 f"infected_A1: "
                 f"{len(model.agents.select(model.agents.condition == 1))}"
                 f"  infected_A2: "
                 f"{len(model.agents.select(model.agents.condition == 2))}\n")
    
    
    
def plot_model(parameters):
    fig, axs = plt.subplots(1,2,figsize=(20,10))
    model = HIVmodel(parameters)
    anim = ap.animate(model, fig, axs, animation_plot)

    return anim

def experiment(parameters,i):
    """ Run a single experiment. """
    model = HIVmodel(parameters)
    result = model.run()
    result.save(exp_name="HIVmodel", exp_id=i)