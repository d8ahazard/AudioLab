from gradio.components.base import Component


class ArgHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ArgHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "args"):
            self.args = {}

    def register_element(self, wrapper_name: str, elem_name: str, gradio_element: Component):
        # Initialize wrapper key in the dictionary
        if wrapper_name not in self.args:
            self.args[wrapper_name] = {}

        # Get initial value (if available)
        element_value = getattr(gradio_element, "value", None)
        print(f"Registered {wrapper_name}.{elem_name} -> {element_value}")
        self.args[wrapper_name][elem_name] = element_value

        # Set listeners for the element
        for method in ["upload", "change", "clear"]:
            if hasattr(gradio_element, method):
                getattr(gradio_element, method)(
                    lambda value, wn=wrapper_name, en=elem_name: self.update_element(wn, en, value),
                    inputs=gradio_element,
                    show_progress="hidden"
                )

    def update_element(self, wrapper_name: str, elem_name: str, value):
        # Dynamically update the dictionary with new values
        if wrapper_name in self.args and elem_name in self.args[wrapper_name]:
            self.args[wrapper_name][elem_name] = value
            print(f"Updated {wrapper_name}.{elem_name} -> {value}")

    def get_args(self):
        return self.args
