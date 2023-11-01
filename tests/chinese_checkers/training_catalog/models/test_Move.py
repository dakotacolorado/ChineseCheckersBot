import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.chinese_checkers.training_catalog import Base
from src.chinese_checkers.training_catalog.models.Move import Move, TABLE_NAME

DATABASE_URL = "sqlite:///:memory:"


class TestMove(unittest.TestCase):

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

    def test_move_insertion_and_query(self):
        move = Move(turn=1, from_position=5, to_position=6, simulation_id=1)
        self.session.add(move)
        self.session.commit()

        fetched_move = self.session.query(Move).filter_by(turn=1).first()

        self.assertIsNotNone(fetched_move, "Move was not inserted into the database.")
        self.assertEqual(fetched_move.from_position, 5, "From position did not match expected value.")
        self.assertEqual(fetched_move.to_position, 6, "To position did not match expected value.")
        self.assertEqual(fetched_move.__repr__(),
                         "<Move(id=1, turn=1, from_position=5, to_position=6, simulation_id=1)>",
                         "Move representation is not as expected.")

    def test_move_meta(self):
        self.assertEqual(Move.Meta.table_name, TABLE_NAME, "Table name does not match.")
        self.assertEqual(Move.Meta.id_column, f"{TABLE_NAME}.id", "ID column does not match.")
