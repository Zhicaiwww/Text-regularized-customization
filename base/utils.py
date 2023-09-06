from inspect import isfunction

DATA_ROOT = 'dataset/data'
OUTPUT_ROOT = 'outputs/'
DEBUG_OUTPUT_ROOT = 'outputs/debug'
PLACE_HODLER = '<krk1>'

LORA_CKPT_TEMPLATE = 'checkpoints/{}/{}/decay_lr_5e-4_gas_4/lora_weight_s2000.safetensors'

DEBUG_DIRS = [ "berry_bowl","shiny_sneaker","backpack","cat","can","robot_toy","dog"]

OBJECT_CLASSES = {
    "barn" : "barn",
    "chair" : "chair",
    "flower" : "flower",
    "table" : "table",
    "wooden_pot" : "pot",
    "backpack" : "backpack",
    "backpack_dog" : "backpack",
    "berry_bowl" : "bowl",
    "can" : "can",
    "candle" : "candle",
    "clock" : "clock",
    "colorful_sneaker" : "sneaker",
    "fancy_boot" : "boot",
    "pink_sunglasses" : "glasses",
    "poop_emoji" : "toy",
    "rc_car" : "toy",
    "robot_toy" : "toy",
    "teapot" : "teapot",
    "shiny_sneaker" : "sneaker",
    "vase" : "vase",
}

LIVE_SUBJECT_CLASSES = {
    "cat" : "cat",
    "cat1" : "cat",
    "cat2" : "cat",
    "dog" : "dog",
    "dog1" : "dog",
    "dog2" : "dog",
    "dog3" : "dog",
    "dog5" : "dog",
    "dog6" : "dog",
    "dog7" : "dog",
    "dog8" : "dog",
    "duck_toy" : "toy",
    "teddybear": "teddybear",
    "tortoise_plushy" : "plushy",
    "bear_plushie" : "stuffed animal",
    "grey_sloth_plushie" : "stuffed animal",
    "monster_toy" : "toy",
    "red_cartoon" : "cartoon",
    "wolf_plushie" : "stuffed animal",
}

OBJECT_PROMPT_LIST = [
'a <placeholder> in the jungle',
'a <placeholder> in the snow',
'a <placeholder> on the beach',
'a <placeholder> on a cobblestone street',
'a <placeholder> on top of pink fabric',
'a <placeholder> on top of a wooden floor',
'a <placeholder> with a city in the background',
'a <placeholder> with a mountain in the background',
'a <placeholder> with a blue house in the background',
'a <placeholder> on top of a purple rug in a forest',
'a <placeholder> wearing a red hat',
'a <placeholder> wearing a santa hat',
'a <placeholder> wearing a rainbow scarf',
'a <placeholder> wearing a black top hat and a monocle',
'a <placeholder> in a chef outfit',
'a <placeholder> in a firefighter outfit',
'a <placeholder> in a police outfit',
'a <placeholder> wearing pink glasses',
'a <placeholder> wearing a yellow shirt',
'a <placeholder> in a purple wizard outfit',
'a red <placeholder>',
'a purple <placeholder>',
'a shiny <placeholder>',
'a wet <placeholder>',
'a cube shaped <placeholder>'
]

LIVE_SUBJECT_PROMPT_LIST = [
'a <placeholder> in the jungle',
'a <placeholder> in the snow',
'a <placeholder> on the beach',
'a <placeholder> on a cobblestone street',
'a <placeholder> on top of pink fabric',
'a <placeholder> on top of a wooden floor',
'a <placeholder> with a city in the background',
'a <placeholder> with a mountain in the background',
'a <placeholder> with a blue house in the background',
'a <placeholder> on top of a purple rug in a forest',
'a <placeholder> with a wheat field in the background',
'a <placeholder> with a tree and autumn leaves in the background',
'a <placeholder> with the Eiffel Tower in the background',
'a <placeholder> floating on top of water',
'a <placeholder> floating in an ocean of milk',
'a <placeholder> on top of green grass with sunflowers around it',
'a <placeholder> on top of a mirror',
'a <placeholder> on top of the sidewalk in a crowded street',
'a <placeholder> on top of a dirt road',
'a <placeholder> on top of a white rug',
'a red <placeholder>',
'a purple <placeholder>',
'a shiny <placeholder>',
'a wet <placeholder>',
'a cube shaped <placeholder>'
]

def parse_templates_class_name(dir_name):
    
    if dir_name in LIVE_SUBJECT_CLASSES.keys():
        templates = LIVE_SUBJECT_PROMPT_LIST
        class_name = LIVE_SUBJECT_CLASSES[dir_name]
        return templates, class_name
    elif dir_name in OBJECT_CLASSES.keys():
        templates = OBJECT_PROMPT_LIST
        class_name = OBJECT_CLASSES[dir_name]
        return templates, class_name
    else:
        raise ValueError(f'concept path name {dir_name} not found')

def is_object_class(class_name):
    return class_name in OBJECT_CLASSES.values()

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d