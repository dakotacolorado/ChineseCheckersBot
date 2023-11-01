import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.chinese_checkers.training_catalog import Base
from src.chinese_checkers.training_catalog.models.GamePlayingModel import GamePlayingModel, TABLE_NAME

DATABASE_URL = "sqlite:///:memory:"


class TestGamePlayingModel(unittest.TestCase):

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

    def test_game_playing_model_insertion_and_query(self):
        model = GamePlayingModel(name="MyModel", version="1.0")
        self.session.add(model)
        self.session.commit()

        fetched_model = self.session.query(GamePlayingModel).filter_by(name="MyModel").first()

        self.assertIsNotNone(fetched_model, "Model was not inserted into the database.")
        self.assertEqual(fetched_model.version, "1.0", "Model version did not match expected value.")
        self.assertEqual(fetched_model.__repr__(), "<GamePlayingModel(id=1, name='MyModel', version='1.0')>",
                         "Model representation is not as expected.")

    def test_game_playing_model_meta(self):
        self.assertEqual(GamePlayingModel.Meta.table_name, TABLE_NAME, "Table name does not match.")
        self.assertEqual(GamePlayingModel.Meta.id_column, f"{TABLE_NAME}.id", "ID column does not match.")
