import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.chinese_checkers.training_catalog import Base
from src.chinese_checkers.training_catalog.models.Simulation import Simulation
from src.chinese_checkers.training_catalog.models.GamePlayingModel import GamePlayingModel, \
    TABLE_NAME as MODEL_TABLE_NAME
from src.chinese_checkers.training_catalog.models.Player import Player
from src.chinese_checkers.training_catalog.models.SimulationModelPlayerAssociation import \
    SimulationModelPlayerAssociation, TABLE_NAME

DATABASE_URL = "sqlite:///:memory:"


class TestSimulationModelPlayerAssociation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(cls.engine)
        cls.Session = sessionmaker(bind=cls.engine)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(cls.engine)

    def setUp(self):
        self.session = self.Session()

    def tearDown(self):
        self.session.close()

    def test_association_insertion_and_query(self):
        # Mocked data for the association.
        player = Player(board_size=121, start_positions="1,2,3", target_positions="119,120,121")
        self.session.add(player)
        self.session.commit()  # Commit the player to the database to get an ID assigned.

        model = GamePlayingModel(name="TestModel", version="1.0")
        simulation = Simulation(number_of_turns=50, number_of_players=2, winning_player_id=player.id, board_size=121)

        self.session.add_all([model, simulation])
        self.session.commit()

        association = SimulationModelPlayerAssociation(simulation_id=simulation.id, game_playing_model_id=model.id,
                                                       player_id=player.id)
        self.session.add(association)
        self.session.commit()

        fetched_assoc = self.session.query(SimulationModelPlayerAssociation).filter_by(
            simulation_id=simulation.id).first()

        self.assertIsNotNone(fetched_assoc, "Association was not inserted into the database.")
        self.assertEqual(fetched_assoc.game_playing_model_id, model.id, "Model ID did not match expected value.")
        self.assertEqual(fetched_assoc.player_id, player.id, "Player ID did not match expected value.")
        self.assertEqual(fetched_assoc.__repr__(),
                         f"<SimulationModelPlayerAssociation(id=1, simulation_id={simulation.id}, game_playing_model_id={model.id}, player_id={player.id})>",
                         "Association representation is not as expected.")

    def test_association_meta(self):
        self.assertEqual(SimulationModelPlayerAssociation.Meta.table_name, TABLE_NAME, "Table name does not match.")
        self.assertEqual(SimulationModelPlayerAssociation.Meta.id_column, f"{TABLE_NAME}.id",
                         "ID column does not match.")
