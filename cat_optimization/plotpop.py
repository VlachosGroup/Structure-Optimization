import matplotlib.pyplot as plt

def plot_pop_MO(data, fname = None, title = None):
        
    '''
    Plot the population fitnesses for a multiple objective genetic algorithm
    
    :param data: 2-column matrix with objective values for each individual in each row
    
    :param fname: File name to save the graph as. If none is given, it just displays.
    
    :param title: Title for the graph.
    '''    
    
    plt.plot(data[:,0], data[:,1], marker='o', color = 'k', linestyle = 'None')       # population
    plt.xlabel('Surface energy (J/m^2)', size=24)
    plt.ylabel('Curent density (mA/cm^2)', size=24)
    plt.xlim([1.2, 2.4])
    plt.ylim([0, 100])
    plt.xticks(size=20)
    plt.yticks(size=20)
    if not title is None:
        plt.title(title, size=24)
    plt.tight_layout()
    
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()
            
            
def plot_pop_SO(data, fname = None, title = None):
        
    '''
    Plot the population fitnesses for a single objective genetic algorithm
    
    :param data: List or 1-D array of values
    
    :param fname: File name to save the graph as. If none is given, it just displays.
    
    :param title: Title for the graph.
    '''    
    
    #plt.hist(data, bins = [i * 5 for i in range(21)])
    plt.hist(data, bins = 50)
    plt.xlabel('Curent density (mA/cm^2)', size=24)
    plt.ylabel('Frequency', size=24)
    #plt.xlim([0, 100])
    #plt.ylim([0, 300])
    plt.xticks(size=20)
    plt.yticks(size=20)
    if not title is None:
        plt.title(title, size=24)
    plt.tight_layout()
    
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()