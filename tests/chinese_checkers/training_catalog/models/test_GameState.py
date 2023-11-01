import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.chinese_checkers.training_catalog import Base
from src.chinese_checkers.training_catalog.models.GameState import GameState, TABLE_NAME

DATABASE_URL = "sqlite:///:memory:"


class TestGameState(unittest.TestCase):

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

    def test_game_state_insertion_and_query(self):
        state = GameState(board_hash="somehash", player_score=100)
        self.session.add(state)
        self.session.commit()

        fetched_state = self.session.query(GameState).filter_by(board_hash="somehash").first()

        self.assertIsNotNone(fetched_state, "GameState was not inserted into the database.")
        self.assertEqual(fetched_state.player_score, 100, "Player score did not match expected value.")
        self.assertEqual(fetched_state.__repr__(), "<GameState(id=1, board_hash='somehash', player_score=100)>",
                         "GameState representation is not as expected.")

    def test_game_state_meta(self):
        self.assertEqual(GameState.Meta.table_name, TABLE_NAME, "Table name does not match.")
        self.assertEqual(GameState.Meta.id_column, f"{TABLE_NAME}.id", "ID column does not match.")
