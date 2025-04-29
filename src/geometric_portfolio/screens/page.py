from abc import abstractmethod


class Page:
    """
    Base class for pages.
    """

    @abstractmethod
    def render(self):
        """
        Render the page.
        """
