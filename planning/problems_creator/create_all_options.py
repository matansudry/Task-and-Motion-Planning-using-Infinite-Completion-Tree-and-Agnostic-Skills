import sys
sys.path.append(".")

from utils.general_utils import save_pickle

def _check_all_not_none(assignment:dict):
    for key in assignment:
        if assignment[key] is None:
            return False
    return True

def has_loop(key:str, assignment:dict):
    temp_key = key
    for _ in range(10):
        if assignment[temp_key] not in assignment.keys():
            return False
        if assignment[key] == key:
            return True
        temp_key = assignment[temp_key]
    return True

def aassignment_has_loop(assignment:dict):
    for key in assignment:
        if has_loop(key=key, assignment=assignment):
            return True
    return False

def assignment_to_himself(assignment):
    for key in assignment:
        if assignment[key] == key:
            return True
    return False

def two_objects_on_the_same_objects(assignment):
    for key in assignment:
        for second_key in assignment:
            if key == second_key:
                continue
            if assignment[key] == assignment[second_key] and key in assignment:
                return True
    return False


if __name__ == "__main__":
    objects = ["red", "yellow", "cyan", "blue"]
    floors = ["red_box", "yellow_box", "cyan_box", "blue_box", "rack", "table"]
    assignment = {
        "red_box":None,
        "yellow_box":None,
        "cyan_box":None,
        "blue_box":None,
    }
    approved_assignments = []
    for red_floor in floors:
        for yellow_floor in floors:
            for cyan_floor in floors:
                for blue_floor in floors:
                    assignment = {
                        "red_box":red_floor,
                        "yellow_box":yellow_floor,
                        "cyan_box":cyan_floor,
                        "blue_box":blue_floor,
                    }
                    if assignment in approved_assignments:
                        continue
                    #check all assignments 
                    if not _check_all_not_none(assignment=assignment):
                        continue
                    if assignment_to_himself(assignment=assignment):
                        continue
                    if aassignment_has_loop(assignment=assignment):
                        continue
                    if two_objects_on_the_same_objects(assignment):
                        continue
                    approved_assignments.append(assignment)
    save_pickle(
        path="all_options.pickle",
        object_to_save=approved_assignments,
    )

