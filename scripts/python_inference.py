from basic_pitch.inference import predict, predict_and_save
import argparse
import os
import sys
import onnx


def print_model_info(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Print input info
    print("Inputs:")
    for input in model.graph.input:
        input_shape = [(dim.dim_value if dim.dim_value != 0 else 'None') for dim in input.type.tensor_type.shape.dim]
        print(f"Name: {input.name}, Shape: {input_shape}")

    # Print output info
    print("\nOutputs:")
    for output in model.graph.output:
        output_shape = [(dim.dim_value if dim.dim_value != 0 else 'None') for dim in output.type.tensor_type.shape.dim]
        print(f"Name: {output.name}, Shape: {output_shape}")


if __name__ == '__main__':
    # set up argparse with input wav file as positional argument
    parser = argparse.ArgumentParser(description='Basic-pitch')
    parser.add_argument('--model-info', action='store_true', help='print model info')
    parser.add_argument('input_file', type=str, help='path to input wav file')
    parser.add_argument('--dest-dir', type=str, default=None, help='path to write output files')

    args = parser.parse_args()

    model_path = os.path.abspath("./ort-model/model.onnx")

    if args.model_info:
        print_model_info(model_path)
        sys.exit(0)

    print(f"Using model: {model_path}")

    if args.dest_dir is not None:
        print(f"Writing MIDI outputs to {args.dest_dir}")
        predict_and_save(
            [args.input_file],
            args.dest_dir,
            True,
            True,
            False,
            False,
            model_or_model_path=model_path,
        )
    else:
        print("No dest dir specified, simply running inference...")
        model_output, midi_data, note_events = predict(
            args.input_file,
            model_or_model_path=model_path,
        )
