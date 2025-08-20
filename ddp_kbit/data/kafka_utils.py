"""
Kafka utility functions for distributed data processing.

This module contains utilities for:
- Offset splitting and partitioning logic
- Kafka partition validation
- Dynamic URL creation for API requests
- Configuration validation and correction
"""

import os
import typing
from urllib.parse import urlencode
from kafka import KafkaAdminClient
from kafka.errors import KafkaError
import torch.distributed as dist


def split_offsets(offsets_data: typing.List[str], split_config: typing.Optional[typing.List[dict]]) -> typing.List[typing.List[tuple]]:
    """
    Split offset data according to the given split configuration.
    
    Args:
        offsets_data (List[str]): List of offset strings in format "partition_id:start_offset:end_offset"
        split_config (Optional[List[dict]]): Configuration for splitting data
    
    Returns:
        List[List[tuple]]: List of partition ranges for each split
    """
    split_config = validate_split_config(split_config)
    offset_diffs = parse_offset_data(offsets_data)
    final_offset_ranges = []

    for i, config in enumerate(split_config):
        is_last_config = (i == len(split_config) - 1)
        partition_ranges = calculate_partition_ranges(offset_diffs, config['rate'], config['fillup'], is_last_config)
        final_offset_ranges.append(partition_ranges)
        offset_diffs = update_offset_diffs(offset_diffs, partition_ranges)

    return final_offset_ranges


def validate_split_config(split_config: typing.Optional[typing.List[dict]]) -> typing.List[dict]:
    """
    Validate and correct split configuration.

    Constraints and settings:
    1. split_config is a list of dictionaries
    2. Each dictionary can have 'rate' and 'fillup' keys
    3. If 'rate' is missing or None:
       - ValueError for non-last items
       - Automatic setting to remaining rate for last item
    4. If 'fillup' is missing or None, automatically set to True
    5. ValueError if intermediate rates sum exceeds 1.0
    6. Add new item if all rates sum < 1.0
    7. Warning if final rate sum != 1.0

    Corrections:
    - Set missing 'rate' in last item to remaining rate
    - Set missing 'fillup' to True
    - Add new item with remaining rate if needed

    Args:
        split_config (Optional[List[dict]]): Configuration to validate and correct
    
    Returns:
        List[dict]: Validated and corrected configuration
    
    Raises:
        ValueError: When invalid configuration is found
    """

    # Add default entry if empty or None
    if not split_config:
        split_config = [{"rate": 1.0, "fillup": True}]
        return split_config
    
    # Calculate total rate sum
    rate_sum = 0.0
    last_index = len(split_config) - 1

    for i, config in enumerate(split_config):
        if "rate" not in config or config["rate"] is None:
            if i == last_index:
                remaining_rate = round(1.0 - rate_sum, 10)
                config["rate"] = remaining_rate
                rate_sum += remaining_rate
            else:
                raise ValueError(f"Error: The 'rate' key is missing or null in split configuration entry {i}, but it is required.")
        else:
            rate_sum += config["rate"]
            if rate_sum > 1.0 and i != last_index:
                raise ValueError(f"Error: The sum of rates exceeds 1.0 after entry {i}. Please check the provided rates.")

        if "fillup" not in config or config["fillup"] is None:
            config["fillup"] = True

    # Add new entry if sum < 1.0
    if rate_sum < 1.0 and not ("rate" in split_config[last_index] and round(rate_sum, 10) == 1.0):
        remaining_rate = round(1.0 - rate_sum, 10)
        split_config.append({"rate": remaining_rate, "fillup": True})

    final_rate_sum = round(sum(config["rate"] for config in split_config), 10)
    if final_rate_sum != 1.0:
        raise ValueError(f"Error: The sum of rates is {final_rate_sum}, which is not 1.0.")

    return split_config


def parse_offset_data(offsets_data: typing.List[str]) -> typing.List[tuple]:
    """
    Parse offset data and calculate offset_diff for each partition.

    Input:
    offsets_data: ["2:0:5833", "11:0:5832", "0:0:5833", ...]

    Output:
    [
        ("2", 0, 5833, 5832),
        ("11", 0, 5832, 5831),
        ("0", 0, 5833, 5832),
        ...
    ]

    Each tuple:
    - partition_id: Partition ID (string)
    - start_offset: Start offset (integer)
    - end_offset: End offset (integer)
    - offset_diff: Difference between start and end offset (integer)
    
    Args:
        offsets_data (List[str]): List of offset strings
    
    Returns:
        List[tuple]: Parsed offset information
    """
    offset_diffs = []
    for offset_info in offsets_data:
        partition_id, start_offset, end_offset = offset_info.split(':')
        offset_diff = int(end_offset) - int(start_offset) - 1
        offset_diffs.append((partition_id, int(start_offset), int(end_offset), offset_diff))
    return offset_diffs


def calculate_partition_ranges(
    offset_diffs: typing.List[tuple],
    rate: float,
    fillup: bool,
    is_last_config: bool
) -> typing.List[tuple]:
    """
    Calculate partition ranges according to the given configuration.

    Input:
    offset_diffs: [
        ("2", 0, 5833, 5832),
        ("11", 0, 5832, 5831),
        ("0", 0, 5833, 5832),
        ...
    ]
    rate: 0.5 (rate to allocate to each partition)
    fillup: True (use maximum available range)
    is_last_config: False (whether this is the last configuration)

    Output:
    partition_ranges: [
        ("2", 0, 2916),
        ("11", 0, 2916),
        ("0", 0, 2916),
        ...
    ]

    - `fillup=True`, `is_last_config=False`:
        - Allocate maximum possible range from each partition's offset.
        - Example: ("2", 0, 5833, 5832) -> ("2", 0, 2916)

    - `fillup=False`, `is_last_config=False`:
        - Divide each partition's offset according to the given rate.
        - Example: ("2", 0, 5833, 5832) -> ("2", 0, 2916) (with `rate=0.5`)

    - `is_last_config=True`:
        - Keep each partition's offset in original range.
        - Example: ("2", 0, 5833, 5832) -> ("2", 0, 5833)
    
    Args:
        offset_diffs (List[tuple]): Parsed offset data
        rate (float): Split rate
        fillup (bool): Whether to fill up to maximum range
        is_last_config (bool): Whether this is the last configuration
    
    Returns:
        List[tuple]: Calculated partition ranges
    """
    partition_ranges = []
    if not is_last_config and fillup:
        max_split_offset_diff = max(max(1, round(od[3] * rate)) for od in offset_diffs)
        for partition_id, start_offset, end_offset, _ in offset_diffs:
            new_end_offset = min(start_offset + max_split_offset_diff, end_offset)
            partition_ranges.append((partition_id, start_offset, new_end_offset))
    elif not is_last_config and not fillup:
        for partition_id, start_offset, end_offset, offset_diff in offset_diffs:
            split_offset_diff = max(1, round(offset_diff * rate))
            new_end_offset = min(start_offset + split_offset_diff, end_offset)
            partition_ranges.append((partition_id, start_offset, new_end_offset))
    else:  # is_last_config
        partition_ranges = [(p_id, start, end) for p_id, start, end, _ in offset_diffs]
    return partition_ranges


def update_offset_diffs(
    offset_diffs: typing.List[tuple],
    partition_ranges: typing.List[tuple]
) -> typing.List[tuple]:
    """
    Update offset_diffs for the next split_config.

    Input:
    offset_diffs: [
        ("2", 0, 5833, 5832),
        ("11", 0, 5832, 5831),
        ("0", 0, 5833, 5832),
        ...
    ]
    partition_ranges: [
        ("2", 0, 2916),
        ("11", 0, 2916),
        ("0", 0, 2916),
        ...
    ]

    Output:
    updated_offset_diffs: [
        ("2", 2917, 5833, 5832),
        ("11", 2917, 5832, 5831),
        ("0", 2917, 5833, 5832),
        ...
    ]

    - `offset_diffs` contains tuples with partition ID, start offset, end offset, and offset difference
    - `partition_ranges` contains tuples with new start and end offsets for each partition
    - Function calculates `new_start_offset` according to `partition_ranges`
    - New start offset begins after the previous end offset
    - Returns updated `offset_diffs` with updated `start_offset`
    
    Args:
        offset_diffs (List[tuple]): Current offset data
        partition_ranges (List[tuple]): New partition ranges
    
    Returns:
        List[tuple]: Updated offset data
    """
    updated_offset_diffs = []
    for (partition_id, _, end_offset, offset_diff), (_, _, new_end) in zip(offset_diffs, partition_ranges):
        new_start_offset = min(new_end + 1, end_offset)
        updated_offset_diffs.append((partition_id, new_start_offset, end_offset, offset_diff))
    return updated_offset_diffs


def print_results(final_offset_ranges: typing.List[typing.List[tuple]]) -> None:
    """
    Print the final results.
    
    Args:
        final_offset_ranges (List[List[tuple]]): Final offset ranges for all splits
    """
    for i, partition_ranges in enumerate(final_offset_ranges):
        print(f"Split Config {i}:")
        for partition_id, start_offset, end_offset in partition_ranges:
            print(f"    Partition {partition_id}: {start_offset} - {end_offset}")


def create_dynamic_url(api_config):
    """
    Create a dynamic URL based on the given parameter dictionary.

    Args:
        api_config (dict): Configuration containing base_url, endpoint, and params
            - base_url (str): Base URL
            - endpoint (str): Endpoint path
            - params (dict): Query parameters dictionary

    Returns:
        str: Dynamically generated URL
    """
    query_string = urlencode(api_config["params"])
    url = f"{api_config['base_url']}/{api_config['endpoint']}/?{query_string}"
    return url


def check_partitions(broker_address, topic):
    """
    Check if the number of partitions in a Kafka topic matches the WORLD_SIZE.

    Args:
        broker_address (str or list): The address(es) of the Kafka broker(s)
        topic (str): The name of the Kafka topic

    Returns:
        bool: True if the number of partitions matches the WORLD_SIZE, False otherwise
    """
    try:
        # Connect to the Kafka broker
        admin_client = KafkaAdminClient(bootstrap_servers=broker_address)

        # Retrieve metadata for the specified topic
        topic_metadata = admin_client.describe_topics([topic])
        topic_info = topic_metadata[0]

        # Get the number of partitions for the topic
        num_partitions = len(topic_info['partitions'])

        # Retrieve WORLD_SIZE from environment variables
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        print(f"Topic '{topic}' has {num_partitions} partitions. WORLD_SIZE is {world_size}.")

        # Compare the number of partitions with WORLD_SIZE
        if num_partitions == world_size:
            print("The number of partitions matches the WORLD_SIZE.")
            return True
        else:
            print("The number of partitions does not match the WORLD_SIZE.")
            return False

    except KafkaError as e:
        print(f"Kafka error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_kafka_topic_info(broker_address, topic):
    """
    Get detailed information about a Kafka topic.
    
    Args:
        broker_address (str or list): Kafka broker address(es)
        topic (str): Topic name
    
    Returns:
        dict: Topic information including partitions, replicas, etc.
    """
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=broker_address)
        topic_metadata = admin_client.describe_topics([topic])
        
        if topic_metadata:
            return topic_metadata[0]
        else:
            return None
            
    except Exception as e:
        print(f"Error getting topic info: {e}")
        return None


def validate_kafka_connection(broker_address):
    """
    Validate connection to Kafka broker.
    
    Args:
        broker_address (str or list): Kafka broker address(es)
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=broker_address)
        # Try to list topics to test connection
        topics = admin_client.list_topics()
        print(f"Successfully connected to Kafka. Found {len(topics)} topics.")
        return True
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        return False


def estimate_data_distribution(offsets_data, world_size):
    """
    Estimate how data will be distributed across processes.
    
    Args:
        offsets_data (List[str]): List of offset strings
        world_size (int): Number of processes
    
    Returns:
        dict: Distribution statistics
    """
    offset_diffs = parse_offset_data(offsets_data)
    
    total_records = sum(diff[3] + 1 for diff in offset_diffs)  # +1 because diff is end - start - 1
    avg_records_per_partition = total_records / len(offset_diffs)
    avg_records_per_process = total_records / world_size
    
    partition_sizes = [diff[3] + 1 for diff in offset_diffs]
    min_partition_size = min(partition_sizes)
    max_partition_size = max(partition_sizes)
    
    return {
        'total_records': total_records,
        'avg_records_per_partition': avg_records_per_partition,
        'avg_records_per_process': avg_records_per_process,
        'min_partition_size': min_partition_size,
        'max_partition_size': max_partition_size,
        'partition_sizes': partition_sizes
    }