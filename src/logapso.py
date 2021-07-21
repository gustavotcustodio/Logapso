# import random
from pso import Pso
from fitness_function import FitnessFunction


def logapso_ga_func(eval_func, particle):
    def wrapper(chromosome):
        return eval_func(particle + chromosome)
    return wrapper


class Logapso(Pso):

    def __init__(self, swarm_size, inertia, acc1, acc2, max_iters, ga_step_val,
                 fitnessfunction, ga, checkpoint_file=None):

        super().__init__(swarm_size, inertia, acc1, acc2, max_iters,
                         fitnessfunction, checkpoint_file)
        self.ga_step_val = ga_step_val
        self.ga = ga

    # step_value
    def apply_ga(self, particle):
        self.ga.run_ga()
        # ga.run_ga()
        return 0

    def run_pso(self):
        # probability_ga = 0.1  # Probability to run GA t fine-tune solution

        for _ in range(self.max_iters):
            for particle in self.particles:
                particle.update_velocity(self.best_particle.position,
                                         self.inertia, self.acc1, self.acc2)
                particle.update_position(self.fitnessfunction)
                particle.set_current_fitness(
                    self.fitnessfunction.calc_fitness(particle.position))

                # Update the best position found by the particle
                if self.fitnessfunction.is_fitness_improved(
                    particle.current_fitness, particle.best_fitness
                ):
                    particle.update_best_position()
                    particle.set_best_fitness(particle.current_fitness)

                    # if random.random(0, 1) < probability_ga:
                    #     print('aplicou o GA no local')
                    #     pnew <- applyGA(pi) // GA is applied 10% of times
                    #     if f(pnew) < f(pi) then
                    #         pi,xi <- pnew

                    # Update the global solution if it is a better candidate
                    self._update_best_particle(particle)

            if self.updated_global_solution:
                print('aplicou o GA no global')
            #     gnew <- applyGA(g)// applies the GA on global solution
            #     if f(gnew)< f(g) then
            #         g,pbest,xbest <- gnew// if solution improves update

            print(self.best_particle.best_fitness)
            self.updated_global_solution = False


def sum_all(arr):
    return sum(arr)


if __name__ == '__main__':
    fitnessfunction = FitnessFunction(sum_all, maximization=True)
    pso_optimizer = Logapso(50, 0.1, 0.2, 0.2, 300, 0.2, fitnessfunction)
    pso_optimizer.generate_particles(12, lbound=-1, ubound=1)
    pso_optimizer.run_pso()
