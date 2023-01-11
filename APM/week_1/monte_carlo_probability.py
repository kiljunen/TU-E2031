from pert import PERT
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from statistics import mean
from random import randint

"""
SETUP
"""
sample_size = 10000
target_task_id = "A11"
speed_up_option_names = ["speed_up_A7", "divide_A10", "guarantee_A4_A5_A9"]
probabilities_for_double_work = {"A4": 0.3}

"""
FUNCTIONS
"""


def get_model_attributes(
    speed_up_A7: bool,
    divide_A10: bool,
    guarantee_A4_A5_A9: bool,
) -> tuple[dict, dict, dict]:
    if speed_up_A7:
        speed_up_A7_factor = 6
        A7_additional_cost = 20000
    else:
        speed_up_A7_factor = 0
        A7_additional_cost = 0

    A7_times = (
        15 - speed_up_A7_factor,
        20 - speed_up_A7_factor,
        25 - speed_up_A7_factor,
    )

    delta = 0.1

    if divide_A10:
        A10_dependencies = {
            "a10": ["A8"],
            "b10": ["a10", "A9"],
        }
        A10_times = {
            "a10": (4 - delta, 4, 4 + delta),
            "b10": (4 - delta, 4, 4 + delta),
        }
        _A10_total_cost = 45000 + 15000
        A10_cost_estimate = {
            "a10": _A10_total_cost / 2,
            "b10": _A10_total_cost / 2,
        }
        final_A10_id = "b10"
    else:
        A10_dependencies = {
            "A10": ["A8", "A9"],
        }
        A10_times = {
            "A10": (8, 10, 12),
        }
        A10_cost_estimate = {
            "A10": 45000,
        }
        final_A10_id = "A10"

    if guarantee_A4_A5_A9:
        A4_A5_A9_task_times = {
            "A4": (8 - delta, 8, 8 + delta),
            "A5": (7 - delta, 7, 7 + delta),
            "A9": (10 - delta, 10, 10 + delta),
        }
        additional_cost = 18000 / 3
        A4_A5_A9_cost_estimates = {
            "A4": 35000 + additional_cost,
            "A5": 35000 + additional_cost,
            "A9": 45000 + additional_cost,
        }
    else:
        A4_A5_A9_task_times = {
            "A4": (6, 8, 12),
            "A5": (6, 7, 9),
            "A9": (8, 10, 13),
        }
        A4_A5_A9_cost_estimates = {
            "A4": 35000,
            "A5": 35000,
            "A9": 45000,
        }

    task_dependencies = {
        "A1": [],
        "A2": ["A1"],
        "A3": ["A2"],
        "A4": ["A2"],
        "A5": ["A4"],
        "A6": ["A2", "A4"],
        "A7": ["A3", "A6"],
        "A8": ["A3", "A5", "A6"],
        "A9": ["A5"],
        "A11": ["A7", final_A10_id],
    } | A10_dependencies

    task_times = (
        {
            "A1": (2, 3, 4),
            "A2": (4, 5, 6),
            "A3": (10, 12, 15),
            "A6": (5, 6, 8),
            "A7": A7_times,
            "A8": (4, 5, 7),
            "A11": (4, 5, 7),
        }
        | A10_times
        | A4_A5_A9_task_times
    )

    cost_estimate = (
        {
            "A1": 15000,
            "A2": 30000,
            "A3": 80000,
            "A6": 30000,
            "A7": 100000 + A7_additional_cost,
            "A8": 25000,
            "A11": 30000,
        }
        | A10_cost_estimate
        | A4_A5_A9_cost_estimates
    )
    return task_dependencies, task_times, cost_estimate


def get_task_completion_times(
    minimum: int,
    expected: int,
    maximum: int,
    sample_size: int,
    probability_of_double_work: float,
) -> list[float]:
    pert = PERT(min_val=minimum, ml_val=expected, max_val=maximum)
    completion_times = list(pert.rvs(size=sample_size))

    double_completion_times = []
    for completion_time in completion_times:
        random_variable = randint(0, 100) / 100
        is_double_work = random_variable <= probability_of_double_work
        if is_double_work:
            completion_time *= 2
        double_completion_times.append(completion_time)

    return double_completion_times


def calculate_task_total_completion_time(
    target_task_id: str,
    task_dependencies: dict[str, list[str]],
    task_completion_times: dict[str, float],
) -> float:
    completion_times = []
    task_dependency_list = task_dependencies[target_task_id]

    for item_id in task_dependency_list:
        item_completion_time = calculate_task_total_completion_time(
            target_task_id=item_id,
            task_dependencies=task_dependencies,
            task_completion_times=task_completion_times,
        )
        completion_times.append(item_completion_time)

    if len(completion_times) > 0:
        longest_running_task = max(completion_times)
    else:
        longest_running_task = 0

    target_task_completion_time = task_completion_times[target_task_id]
    total_completion_time = longest_running_task + target_task_completion_time
    return total_completion_time


def calculate_total_completion_time_of_target_task(
    target_task_id: str,
    task_times: dict[str, tuple[int, int, int]],
    task_dependencies: dict[str, list[str]],
    sample_size: int,
) -> list[float]:
    completions = {}
    for task_id in task_times:
        minimum, expected, maximum = task_times[task_id]

        if task_id in probabilities_for_double_work:
            probability_of_double_work = probabilities_for_double_work[task_id]
        else:
            probability_of_double_work = 0

        list_of_task_completion_times = get_task_completion_times(
            minimum=minimum,
            expected=expected,
            maximum=maximum,
            sample_size=sample_size,
            probability_of_double_work=probability_of_double_work,
        )
        completions[task_id] = list_of_task_completion_times

    df = DataFrame(completions)

    total_completion_times = []
    for _, row in df.iterrows():
        task_completion_times = dict(row)

        total_completion_time = calculate_task_total_completion_time(
            target_task_id=target_task_id,
            task_dependencies=task_dependencies,
            task_completion_times=task_completion_times,
        )

        total_completion_times.append(total_completion_time)

    return total_completion_times


"""
CALCULATE
"""

all_combinations = list(product([True, False], repeat=len(speed_up_option_names)))

all_cases = []
for combination in all_combinations:
    parameters = dict(zip(speed_up_option_names, combination))
    task_dependencies, task_times, cost_estimate = get_model_attributes(
        **parameters
    )
    case = parameters | {
        "task_dependencies": task_dependencies,
        "task_times": task_times,
        "cost_estimate": cost_estimate,
    }
    all_cases.append(case)


calculated_case_kpis = {}
calculated_case_times = {}
for case in all_cases:
    speed_up_A7 = case["speed_up_A7"]
    divide_A10 = case["divide_A10"]
    guarantee_A4_A5_A9 = case["guarantee_A4_A5_A9"]
    task_times = case["task_times"]
    task_dependencies = case["task_dependencies"]
    cost_estimate = case["cost_estimate"]
    total_completion_times = calculate_total_completion_time_of_target_task(
        target_task_id=target_task_id,
        task_times=task_times,
        task_dependencies=task_dependencies,
        sample_size=sample_size,
    )
    total_completion_time_mean = mean(total_completion_times)
    total_cost = sum(cost_estimate.values())
    total_cost_to_time_mean = total_cost / total_completion_time_mean

    case_name = f"{speed_up_A7=}, {divide_A10=}, {guarantee_A4_A5_A9=}"

    calculated_case_kpis[case_name] = {
        "total_cost": total_cost,
        "average_completion_time": total_completion_time_mean,
    }
    calculated_case_times[case_name] = total_completion_times


check = {
    "speed_up_A7=True, divide_A10=True, guarantee_A4_A5_A9=True": {
        "total_cost": 523000.0,
        "average_completion_time": 42.65354352327895,
    },
    "speed_up_A7=True, divide_A10=True, guarantee_A4_A5_A9=False": {
        "total_cost": 505000.0,
        "average_completion_time": 43.351366321125234,
    },
    "speed_up_A7=True, divide_A10=False, guarantee_A4_A5_A9=True": {
        "total_cost": 508000.0,
        "average_completion_time": 48.15591378712529,
    },
    "speed_up_A7=True, divide_A10=False, guarantee_A4_A5_A9=False": {
        "total_cost": 490000,
        "average_completion_time": 48.82171435838333,
    },
    "speed_up_A7=False, divide_A10=True, guarantee_A4_A5_A9=True": {
        "total_cost": 503000.0,
        "average_completion_time": 47.40052383043162,
    },
    "speed_up_A7=False, divide_A10=True, guarantee_A4_A5_A9=False": {
        "total_cost": 485000.0,
        "average_completion_time": 47.68190436302683,
    },
    "speed_up_A7=False, divide_A10=False, guarantee_A4_A5_A9=True": {
        "total_cost": 488000.0,
        "average_completion_time": 48.68337371256089,
    },
    "speed_up_A7=False, divide_A10=False, guarantee_A4_A5_A9=False": {
        "total_cost": 470000,
        "average_completion_time": 49.325575974779575,
    },
}


"""
PLOT
"""

fig, ax = plt.subplots(figsize=(6, 6))
fig.subplots_adjust(bottom=0.4)
ax.set_xlim(38, 57.5)
ax.set_ylim(0, 0.4)

best_indices = [0]

colors = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "violet",
    "brown",
    "black",
]
legend = []
for index, (calculated_case_time, color) in enumerate(
    zip(calculated_case_times, colors)
):
    if index not in best_indices:
        continue
    data = calculated_case_times[calculated_case_time]
    sns.kdeplot(data=data, ax=ax, color=color)
    legend.append(calculated_case_time)

plt.legend(legend, bbox_to_anchor=(1.05, -0.1))
plt.show()
