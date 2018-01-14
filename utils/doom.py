from vizdoom import *
import re
import cv2

def set_doom_configuration(game, params):
    game.set_doom_scenario_path(params.scenario_path)

    if (params.visualize):
        # Use a bigger screen size when visualizing
        game.set_screen_resolution(ScreenResolution.RES_800X450)
    else:
        # Use a smaller screen size for faster simulation
        game.set_screen_resolution(ScreenResolution.RES_400X225)

    game.set_screen_format(ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)

    # sets other rendering options
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # if hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(False)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)

    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.MOVE_FORWARD)

    game.set_episode_timeout(30000)

    # makes episodes start after 10 tics
    game.set_episode_start_time(10)

    # makes the window appear (turned on by default)
    game.set_window_visible(True if params.visualize else False)

    if params.interactive:
        game.set_mode(Mode.SPECTATOR)
    else:
        game.set_mode(Mode.PLAYER)

    return game


def get_doom_coordinates(x, y):
    return int(x) * 256 * 256, int(y) * 256 * 256


def get_world_coordinates(x):
    return x / (256 * 256)


def get_agent_location(game):
    x = get_world_coordinates(game.get_game_variable(GameVariable.USER3))
    y = get_world_coordinates(game.get_game_variable(GameVariable.USER4))
    return x, y


def spawn_object(game, object_id, x, y):
    x_pos, y_pos = get_doom_coordinates(x, y)
    # call spawn function twice because vizdoom objects are not spawned
    # sometimes if spawned only once for some unknown reason
    for _ in range(2):
        game.send_game_command("pukename spawn_object_by_id_and_location \
                                %i %i %i" % (object_id, x_pos, y_pos))
        pause_game(game, 1)


def spawn_agent(game, x, y, orientation):
    x_pos, y_pos = get_doom_coordinates(x, y)
    game.send_game_command("pukename set_position %i %i %i" %
                           (x_pos, y_pos, orientation))


def pause_game(game, steps):
    for i in range(1):
        r = game.make_action([False, False, False])


def split_object(object_string):
    split_word = re.findall('[A-Z][^A-Z]*', object_string)
    split_word.reverse()
    return split_word


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points
    """
    return ((x1-y1)**2 + (x2-y2)**2)**0.5


def process_screen(screen, height, width):
    """
    Resize the screen.
    """
    if screen.shape != (3, height, width):
        screen = cv2.resize(screen, (width, height),
                            interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    return screen


class DoomObject(object):
    def __init__(self, *args):
        self.name = ''.join(list(reversed(args)))
        self.type = args[0]

        try:
            # Bug in Vizdoom, BlueArmor is actually red.
            # I can see your expression ! ;-)
            if self.name == 'BlueArmor':
                self.color = 'Red'
            else:
                self.color = args[1]
        except IndexError:
            self.color = None

        try:
            self.relative_size = args[2]
        except IndexError:
            self.relative_size = None

        try:
            self.absolute_size = args[3]
        except IndexError:
            self.absolute_size = None
