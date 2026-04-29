import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://overload:overload@localhost:5432/overload",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class WorkoutLog(Base):
    __tablename__ = "workout_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    logged_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Exercise
    exercise = Column(String, nullable=False)
    one_rm = Column(Float)

    # Program context
    week = Column(Integer)
    day = Column(Integer)
    program_length = Column(Integer)
    time_per_workout = Column(Float)
    number_of_exercises = Column(Integer)
    weeks_gap = Column(Integer, default=1)

    # Last session (lag features — inputs to prediction)
    lag_sets = Column(Integer)
    lag_reps = Column(Float)
    lag_rpe = Column(Float)

    # This session (actual outcome — used to compute deltas for retraining)
    sets = Column(Integer)
    reps = Column(Float)
    rpe = Column(Float)

    # Fitness level flags
    level_Advanced = Column(Integer, default=0)
    level_Beginner = Column(Integer, default=0)
    level_Intermediate = Column(Integer, default=0)
    level_Novice = Column(Integer, default=0)

    # Goal flags
    goal_at_home_calisthenics = Column(Integer, default=0)
    goal_athletics = Column(Integer, default=0)
    goal_bodybuilding = Column(Integer, default=0)
    goal_bodyweight_fitness = Column(Integer, default=0)
    goal_muscle_sculpting = Column(Integer, default=0)
    goal_olympic_weightlifting = Column(Integer, default=0)
    goal_powerbuilding = Column(Integer, default=0)
    goal_powerlifting = Column(Integer, default=0)

    # Equipment flags
    equipment_at_home = Column(Integer, default=0)
    equipment_dumbbell_only = Column(Integer, default=0)
    equipment_full_gym = Column(Integer, default=0)
    equipment_garage_gym = Column(Integer, default=0)


def create_tables() -> None:
    Base.metadata.create_all(engine)
