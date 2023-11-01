import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.chinese_checkers.training_catalog import Base
from src.chinese_checkers.training_catalog.models.Simulation import Simulation, TABLE_NAME
from src.chinese_checkers.training_catalog.models.Player import Player

DATABASE_URL = "sqlite:///:memory:"


class TestSimulation(unittest.TestCase):

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

    def test_simulation_insertion_and_query(self):
        player = Player(board_size=121, start_positions="1,2,3", target_positions="119,120,121")
        self.session.add(player)
        self.session.commit()

        simulation = Simulation(number_of_turns=50, number_of_players=2, winning_player_id=player.id, board_size=121)
        self.session.add(simulation)
        self.session.commit()

        fetched_simulation = self.session.query(Simulation).filter_by(number_of_turns=50).first()

        self.assertIsNotNone(fetched_simulation, "Simulation was not inserted into the database.")
        self.assertEqual(fetched_simulation.number_of_players, 2, "Number of players did not match expected value.")
        self.assertEqual(fetched_simulation.winning_player_id, player.id,
                         "Winning player ID did not match expected value.")
        self.assertEqual(fetched_simulation.__repr__(),
                         f"<Simulation(id=1, number_of_turns=50, number_of_players=2, winning_player_id={player.id}, board_size=121)>",
                         "Simulation representation is not as expected.")

    def test_simulation_meta(self):
        self.assertEqual(Simulation.Meta.table_name, TABLE_NAME, "Table name does not match.")
        self.assertEqual(Simulation.Meta.id_column, f"{TABLE_NAME}.id", "ID column does not match.")
