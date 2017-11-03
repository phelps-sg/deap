#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every
evolutionary algorithm may vary infinitely. Most of the algorithms in this
module use operators registered in the toolbox. Generally, the keyword used are
:meth:`mate` for crossover, :meth:`mutate` for mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.

You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import random
import logging
import deap.tools

from abc import ABC, abstractmethod


class EvolutionaryAlgorithm(ABC):
    """
    Abstract Base Class (ABC) for all evolutionary algorithms.
    """

    def __init__(self, population, toolbox, cxpb, mutpb, stats, halloffame):
        self.population = population
        self.toolbox = toolbox
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.halloffame = halloffame
        self.stats = stats
        self.logger = logging.getLogger('EvolutionaryAlgorithm')
        self.generation_number = 0
        self.logbook = deap.tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + (self.stats.fields if self.stats else [])

    def varOr(self, lambda_):
        """Part of an evolutionary algorithm applying only the variation part
        (crossover, mutation **or** reproduction). The modified individuals have
        their fitness invalidated. The individuals are cloned so returned
        population is independent of the input population.

        :param lambda\_: The number of children to produce

        The variation goes as follow. On each of the *lambda_* iteration, it
        selects one of the three operations; crossover, mutation or reproduction.
        In the case of a crossover, two individuals are selected at random from
        the parental population :math:`P_\mathrm{p}`, those individuals are cloned
        using the :meth:`toolbox.clone` method and then mated using the
        :meth:`toolbox.mate` method. Only the first child is appended to the
        offspring population :math:`P_\mathrm{o}`, the second child is discarded.
        In the case of a mutation, one individual is selected at random from
        :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
        :meth:`toolbox.mutate` method. The resulting mutant is appended to
        :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
        selected at random from :math:`P_\mathrm{p}`, cloned and appended to
        :math:`P_\mathrm{o}`.

        This variation is named *Or* beceause an offspring will never result from
        both operations crossover and mutation. The sum of both probabilities
        shall be in :math:`[0, 1]`, the reproduction probability is
        1 - *cxpb* - *mutpb*.
        """
        assert (self.cxpb + self.mutpb) <= 1.0, (
            "The sum of the crossover and mutation probabilities must be smaller "
            "or equal to 1.0.")

        offspring = []
        for _ in range(lambda_):
            op_choice = random.random()
            if op_choice < self.cxpb:            # Apply crossover
                ind1, ind2 = map(self.toolbox.clone, random.sample(self.population, 2))
                ind1, ind2 = self.toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                offspring.append(ind1)
            elif op_choice < self.cxpb + self.mutpb:  # Apply mutation
                ind = self.toolbox.clone(random.choice(self.population))
                ind, = self.toolbox.mutate(ind)
                del ind.fitness.values
                offspring.append(ind)
            else:                           # Apply reproduction
                offspring.append(random.choice(self.population))

        assert len(offspring) == lambda_

        return offspring

    def varAnd(self):
        """Part of an evolutionary algorithm applying only the variation part
        (crossover **and** mutation). The modified individuals have their
        fitness invalidated. The individuals are cloned so returned population is
        independent of the input population.

        :returns: A list of varied individuals that are independent of their
                  parents.

        The variation goes as follow. First, the parental population
        :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
        and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
        first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
        consecutive individuals. According to the crossover probability *cxpb*, the
        individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
        using the :meth:`toolbox.mate` method. The resulting children
        :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
        parents in :math:`P_\mathrm{o}`. A second loop over the resulting
        :math:`P_\mathrm{o}` is executed to mutate every individual with a
        probability *mutpb*. When an individual is mutated it replaces its not
        mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
        is returned.

        This variation is named *And* beceause of its propention to apply both
        crossover and mutation on the individuals. Note that both operators are
        not applied systematicaly, the resulting individuals can be generated from
        crossover only, mutation only, crossover and mutation, and reproduction
        according to the given probabilities. Both probabilities should be in
        :math:`[0, 1]`.
        """
        offspring = [self.toolbox.clone(ind) for ind in self.population]

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < self.cxpb:
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1],
                                                              offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < self.mutpb:
                offspring[i], = self.toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        return offspring

    def update_hall_of_fame(self):
        if self.halloffame is not None:
            self.halloffame.update(self.population)

    def update_stats(self, individuals):
        record = self.stats.compile(individuals) if self.stats else {}
        self.logbook.record(gen=self.generation_number, nevals=len(individuals), **record)
        self.logger.debug(self.logbook.stream)

    def evaluate_individuals(self, individuals):
        return self.toolbox.group_and_evaluate(self.toolbox, individuals)

    def individuals_to_evaluate(self, individuals):
        return [i for i in individuals if self.toolbox.fitness_needs_computing(i)]

    def compute_fitnesses(self, individuals):
        for individual, fitness in zip(individuals, self.evaluate_individuals(individuals)):
            individual.fitness.values = fitness
        return individuals

    def evaluate_and_analyse_individuals(self, individuals):
        self.update_stats(self.compute_fitnesses(self.individuals_to_evaluate(individuals)))
        self.update_hall_of_fame()

    def evaluate_and_analyse_population(self):
        self.evaluate_and_analyse_individuals(self.population)

    def prepare_population(self):
        self.evaluate_and_analyse_population()

    def evolve(self, max_generations):
        self.prepare_population()
        for self.generation_number in range(1, max_generations + 1):
            offspring = self.breed()
            self.evaluate_and_analyse_individuals(offspring)
            self.new_generation(offspring)

    @abstractmethod
    def breed(self):
        pass

    @abstractmethod
    def new_generation(self, offspring):
        pass


class SimpleEvolutionaryAlgorithm:
    """
       This algorithm implements the simplest evolutionary algorithm as
       presented in chapter 7 of [Back2000]_.

       :param ngen: The number of generation.

       The algorithm takes in a population and evolves it in place using the
       :meth:`varAnd` method.

       .. note::

           Using a non-stochastic selection method will result in no selection as
           the operator selects *n* individuals from a pool of *n*.

       This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
       :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
       registered in the toolbox.

       .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
          Basic Algorithms and Operators", 2000.
    """

    def breed(self):
        return self.varAnd(self.toolbox.select(self.population, len(self.population)))

    def new_generation(self, offspring):
        self.population[:] = offspring


class EvolutionStrategies(EvolutionaryAlgorithm):
    """
    Abstract superclass of all algorithms implementing evolution strategies.
    """

    def __init__(self, population, toolbox, cxpb, mutpb, stats, halloffame, mu, lambda_):
        EvolutionaryAlgorithm.__init__(self, population, toolbox, cxpb, mutpb, stats, halloffame)
        self.mu = mu
        self.lambda_ = lambda_

    def prepare_population(self):
        pass

    def evaluate_and_analyse_individuals(self, individuals):
        self.compute_fitnesses(self.individuals_to_evaluate(individuals))
        self.update_stats([i for i in self.population if i.fitness.valid])
        self.update_hall_of_fame()

    def breed(self):
        return self.varOr(self.lambda_)

    @abstractmethod
    def new_generation(self, offspring):
        pass


class EvolutionStrategiesMuPlusLambda(EvolutionStrategies):
    """
    This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    """

    def new_generation(self, offspring):
        self.population[:] = self.toolbox.select(self.population + offspring, self.mu)


class EvolutionStrategiesMuCommaLambda(EvolutionStrategies):
    """
    This is the :math:`(\mu~,~\lambda)` evolutionary algorithm.
    """

    def __init__(self, population, toolbox, cxpb, mutpb, stats, halloffame, mu, lambda_):
        assert lambda_ >= mu, "lambda must be greater or equal to mu."
        EvolutionStrategies.__init__(self, population, toolbox, cxpb, mutpb, stats, halloffame, mu, lambda_)

    def new_generation(self, offspring):
        self.population[:] = self.toolbox.select(offspring, self.mu)


# def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
#                      verbose=__debug__):
#     """This is algorithm implements the ask-tell model proposed in
#     [Colette2010]_, where ask is called `generate` and tell is called `update`.
#
#     :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
#                     operators.
#     :param ngen: The number of generation.
#     :param stats: A :class:`~deap.tools.Statistics` object that is updated
#                   inplace, optional.
#     :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
#                        contain the best individuals, optional.
#     :param verbose: Whether or not to log the statistics.
#     :returns: The final population
#     :returns: A class:`~deap.tools.Logbook` with the statistics of the
#               evolution
#
#     The algorithm generates the individuals using the :func:`toolbox.generate`
#     function and updates the generation method with the :func:`toolbox.update`
#     function. It returns the optimized population and a
#     :class:`~deap.tools.Logbook` with the statistics of the evolution. The
#     logbook will contain the generation number, the number of evalutions for
#     each generation and the statistics if a :class:`~deap.tools.Statistics` is
#     given as argument. The pseudocode goes as follow ::
#
#         for g in range(ngen):
#             population = toolbox.generate()
#             evaluate(population)
#             toolbox.update(population)
#
#     .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
#        R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
#        Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
#        Multidisciplinary Design Optimization in Computational Mechanics,
#        Wiley, pp. 527-565;
#
#     """
#     logbook = tools.Logbook()
#     logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
#
#     for gen in xrange(ngen):
#         # Generate a new population
#         population = toolbox.generate()
#         # Evaluate the individuals
#         fitnesses = toolbox.map(toolbox.evaluate, population)
#         for ind, fit in zip(population, fitnesses):
#             ind.fitness.values = fit
#
#         if halloffame is not None:
#             halloffame.update(population)
#
#         # Update the strategy with the evaluated individuals
#         toolbox.update(population)
#
#         record = stats.compile(population) if stats is not None else {}
#         logbook.record(gen=gen, nevals=len(population), **record)
#         if verbose:
#             print logbook.stream
#
#     return population, logbook


def evolutionary_algorithm(*params, max_generations, Algorithm=SimpleEvolutionaryAlgorithm):
    """
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    """
    ea = Algorithm(*params)
    ea.evolve(max_generations)
    return ea.population, ea.logbook


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__):
    return evolutionary_algorithm(population, toolbox, cxpb, mutpb, stats, halloffame,
                                  max_generations=ngen)


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    """
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    """
    return evolutionary_algorithm(population, toolbox, cxpb, mutpb, stats, halloffame, mu, lambda_,
                                  max_generations=ngen, Algorithm=EvolutionStrategiesMuPlusLambda)


def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__):
    """
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    """
    return evolutionary_algorithm(population, toolbox, cxpb, mutpb, stats, halloffame, mu, lambda_,
                                  max_generations=ngen, Algorithm=EvolutionStrategiesMuCommaLambda)

