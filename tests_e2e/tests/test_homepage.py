import pytest
from playwright.sync_api import Page, expect

def test_has_title(page: Page):
    # Tu zmień na adres swojego Flaska, np. jak odpalasz lokalnie na 5000
    page.goto("http://127.0.0.1:5000/")
    expect(page).to_have_title("Rafał Łabędzki")
