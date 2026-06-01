"""
Flask application factory and base configuration.

Initialize the Flask app and register blueprints here.
"""
from flask import Flask

from src.api.predict import create_predict_blueprint


def create_app(model_path: str = 'models/checkpoints/model_final.pth', device: str = 'cpu') -> Flask:
    """
    Create and configure the Flask application.

    Args:
        model_path: Path to the trained model (directory for StarDist or .pth file for UNet).
        device: Device to run inference on (e.g., 'cpu', 'cuda:0').

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)

    # Register prediction blueprint
    predict_bp = create_predict_blueprint(model_path=model_path, device=device)
    app.register_blueprint(predict_bp, url_prefix='/api')

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return {'status': 'ok'}, 200

    return app


if __name__ == '__main__':
    app = create_app(model_path='models/checkpoints/model_final.pth', device='cpu')
    app.run(debug=True, host='0.0.0.0', port=5000)
