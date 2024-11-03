from ..experience.IRewardStrategy import IRewardStrategy
from ..experience.DistanceToWinRewardStrategy import DistanceToWinRewardStrategy


class RewardStrategyFactory:
    """
    Factory for creating instances of IRewardStrategy.

    This factory allows dynamic creation of reward strategies based on the specified
    strategy type. It supports passing additional parameters to customize the reward
    calculation for each strategy type.
    """

    @staticmethod
    def create(strategy_name: str, **kwargs) -> IRewardStrategy:
        """
        Create an instance of the specified reward strategy.

        Args:
            strategy_name (str): The name of the reward strategy to create.
            kwargs: Additional parameters specific to the reward strategy being created.

        Returns:
            IRewardStrategy: An instance of the specified reward strategy.

        Raises:
            ValueError: If an unsupported strategy_name is provided.
        """
        strategies = {
            "distance_to_win": DistanceToWinRewardStrategy,
            # Add other strategies here as they are implemented, e.g.:
            # "another_strategy": AnotherRewardStrategy,
        }

        strategy_class = strategies.get(strategy_name.lower())
        if not strategy_class:
            raise ValueError(f"Unsupported strategy name: {strategy_name}")

        # Initialize the strategy with any additional arguments
        return strategy_class(**kwargs)
