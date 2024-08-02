from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType, HailoSchedulingAlgorithm)
from concurrent.futures import Future
from functools import partial
import numpy as np


class Hailo:
    def __init__(self, hef_path, batch_size=None, output_type='FLOAT32'):
        """
        Initialize the HailoAsyncInference class with the provided HEF model file path.

        Args:
            hef_path (str): Path to the HEF model file.
            batch_size (int): Batch size for inference.
            output_type (str): Format type of the output stream.
        """
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.batch_size = batch_size
        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1 if batch_size is None else batch_size)
        self._set_input_output(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_vstream_info()
        self.configured_infer_model = self.infer_model.configure()

    def __enter__(self):
        """Used for allowing use with context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_traceback):
        """Used for allowing use with context manager."""
        self.close()

    def _set_input_output(self, output_type):
        """
        Set the input and output layer information for the HEF model.

        Args:
            output_type (str): Format type of the output stream.
        """
        input_format_type = self.hef.get_input_vstream_infos()[0].format.type
        self.infer_model.input().set_format_type(input_format_type)
        output_format_type = getattr(FormatType, output_type)
        for output in self.infer_model.outputs:
            output.set_format_type(output_format_type)
        self.num_outputs = len(self.infer_model.outputs)

    def callback(self, completion_info, bindings, future, last):
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the inference task.
            bindings: Bindings object containing input and output buffers.
        """
        if future._has_had_error:
            # Don't really know if this can happen.
            return
        elif completion_info.exception:
            future._has_had_error = True
            future.set_exception(completion_info.exception)
        else:
            # When there's more than one output, we return them all in a dictionary with the layer name,
            # otherwise we merely return the single output on its own.
            if self.num_outputs <= 1:
                this_result = bindings.output().get_buffer()
            else:
                # Don't know why I need an extra expand_dims here. Anyone?
                this_result = {k: np.expand_dims(v.get_buffer(), axis=0) for k, v in bindings._outputs.items()}
            future._intermediate_result.append(this_result)
            if last:
                # On the last callback for this batch, signal to the user that the future is ready.
                if self.batch_size is None:
                    # If the user did not request batches, remove the list that wraps the single result.
                    future._intermediate_result = future._intermediate_result[0]
                future.set_result(future._intermediate_result)

    def _get_vstream_info(self):
        """
        Get information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()

        return input_vstream_info, output_vstream_info

    def get_input_shape(self):
        """
        Get the shape of the model's input layer.

        Returns:
            tuple: Shape of the model's input layer.
        """
        return self.input_vstream_info[0].shape  # Assumes that the model has one input

    def describe(self):
        """
        Return information that describes what's in the model.

        Returns:
            A pair of lists containing, respectively, information about the input and output layers.
        """
        inputs = [(layer.name, layer.shape, layer.format.type) for layer in self.hef.get_input_vstream_infos()]
        outputs = [(layer.name, layer.shape, layer.format.type) for layer in self.hef.get_output_vstream_infos()]

        return inputs, outputs

    def run_async(self, input_data):
        """
        Run asynchronous inference on the Hailo-8 device.

        Args:
            input_data (np.ndarray): Input data for inference.

        Returns:
            future: Future to wait on for the inference results.
        """
        if self.batch_size is None:
            input_data = np.expand_dims(input_data, axis=0)

        future = Future()
        future._intermediate_result = []
        future._has_had_error = False

        for i, frame in enumerate(input_data):
            last = i == len(input_data) - 1
            bindings = self._create_bindings()
            bindings.input().set_buffer(frame)
            self.configured_infer_model.wait_for_async_ready(timeout_ms=10000)
            self.configured_infer_model.run_async([bindings], partial(self.callback, bindings=bindings, future=future, last=last))

        return future

    def run(self, input_data):
        """
        Run asynchronous inference on the Hailo-8 device.

        Args:
            input_data (np.ndarray): Input data for inference.

        Returns:
            inference output or list: Inference output or List of inference outputs if batch_size is not None.
        """
        future = self.run_async(input_data)
        return future.result()

    def _create_bindings(self):
        """
        Create bindings for input and output buffers.

        Returns:
            bindings: Bindings object with input and output buffers.
        """
        output_buffers = {name: np.empty(self.infer_model.output(name).shape, dtype=np.float32)
                          for name in self.infer_model.output_names}
        return self.configured_infer_model.create_bindings(output_buffers=output_buffers)

    def close(self):
        """
        Release the Hailo device.
        """
        del self.configured_infer_model
        self.target.release()
