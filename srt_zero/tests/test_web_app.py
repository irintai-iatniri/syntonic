import unittest
from web_app import app

class TestWebApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SRT Zero', response.data)
        self.assertIn(b'Proton', response.data)

    def test_particle_proton(self):
        response = self.app.get('/particle/proton')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Proton', response.data)
        self.assertIn(b'Tree Level', response.data)
        # Check for some expected values in the HTML (approximate)
        self.assertIn(b'938.272', response.data) 

    def test_particle_electron(self):
        # Electron uses HOOKING_MECHANISM which might fallback or show not available
        # But we should still get a 200 OK page
        response = self.app.get('/particle/electron')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Electron', response.data)

if __name__ == '__main__':
    unittest.main()
