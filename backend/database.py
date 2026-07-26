"""
ExplainX AI — Database layer
100% local persistence using SQLite via SQLAlchemy. No external services.
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./explainx.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ModelRegistry(Base):
    """Every model registered/trained in the platform."""
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    task_type = Column(String)          # classification / regression
    algorithm = Column(String)          # RandomForest, XGBoost, LogisticRegression, ...
    feature_names = Column(JSON)
    target_name = Column(String)
    n_features = Column(Integer)
    n_samples_trained = Column(Integer)
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r2_score = Column(Float, nullable=True)
    model_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExplanationLog(Base):
    """Audit trail of every explanation ever generated — key for compliance."""
    __tablename__ = "explanation_log"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    explanation_type = Column(String)   # global_shap, local_shap, lime, pdp
    input_data = Column(JSON, nullable=True)
    prediction = Column(Float, nullable=True)
    top_features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class FairnessAudit(Base):
    """Bias / fairness evaluation results per protected attribute."""
    __tablename__ = "fairness_audit"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    protected_attribute = Column(String)
    metric_name = Column(String)        # demographic_parity, equal_opportunity, disparate_impact...
    metric_value = Column(Float)
    threshold_passed = Column(String)   # "PASS" / "FAIL" / "WARN"
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
