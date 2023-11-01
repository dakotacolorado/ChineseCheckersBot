import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.chinese_checkers.training_catalog import Base
from src.chinese_checkers.training_catalog.models.Player import Player, TABLE_NAME

DATABASE_URL = "sqlite:///:memory:"


class TestPlayer(unittest.TestCase):

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

    def test_player_insertion_and_query(self):
        player = Player(board_size=121, start_positions="1,2,3", target_positions="119,120,121")
        self.session.add(player)
        self.session.commit()

        fetched_player = self.session.query(Player).filter_by(board_size=121).first()

        self.assertIsNotNone(fetched_player, "Player was not inserted into the database.")
        self.assertEqual(fetched_player.start_positions, "1,2,3", "Start positions did not match expected value.")
        self.assertEqual(fetched_player.target_positions, "119,120,121",
                         "Target positions did not match expected value.")
        self.assertEqual(fetched_player.__repr__(),
                         "<Player(id=1, board_size=121, start_positions='1,2,3', target_positions='119,120,121')>",
                         "Player representation is not as expected.")

    def test_player_meta(self):
        self.assertEqual(Player.Meta.table_name, TABLE_NAME, "Table name does not match.")
        self.assertEqual(Player.Meta.id_column, f"{TABLE_NAME}.id", "ID column does not match.")
