"""How a freshly opened block draws before anyone touches the Columns panel.

The rule has to reproduce exactly what the viewer did on its own, so that
adding the panel changes nothing on screen until a column is switched. These
are the cases the viewer already distinguished: a text file's ``x  y``, a lone
column against the row index, and an HDF5 block that has no such convention.
"""

from src.lib_h5.columns import (
    ColumnRole,
    DisplaySpec,
    Role,
    default_column_roles,
)


def roles_as_tuples(roles):
    return [(c.name, c.role, c.visible) for c in roles]


# ── the defaults ─────────────────────────────────────────────────────── #

def test_one_column_is_a_curve_against_the_row_index():
    roles = default_column_roles(1, ("intensity",))

    assert roles_as_tuples(roles) == [("intensity", Role.Y, True)]
    assert DisplaySpec.from_roles(roles) == DisplaySpec(None, None, ((0, "intensity"),))


def test_a_two_column_text_file_is_one_curve_x_then_y():
    roles = default_column_roles(2, ("energy", "signal"), is_text=True)

    assert roles_as_tuples(roles) == [
        ("energy", Role.X, True),
        ("signal", Role.Y, True),
    ]
    assert DisplaySpec.from_roles(roles) == DisplaySpec(0, "energy", ((1, "signal"),))


def test_a_wide_text_file_keeps_the_first_column_as_x_but_hides_the_curves():
    """It opened as a table; the panel is how a curve joins the plot. The
    abscissa is already the active one, so a curve switched on lands against
    it, not the row index."""
    roles = default_column_roles(4, ("e", "a", "b", "c"), is_text=True)

    assert [c.role for c in roles] == [Role.X, Role.Y, Role.Y, Role.Y]
    assert [c.visible for c in roles] == [True, False, False, False]
    spec = DisplaySpec.from_roles(roles)
    assert spec.has_curves is False
    assert spec.x_index == 0


def test_an_hdf5_block_has_no_x_convention_so_every_column_is_a_curve():
    """This is what the multi-curve view already did: N columns vs the index."""
    roles = default_column_roles(3)

    assert [c.role for c in roles] == [Role.Y, Role.Y, Role.Y]
    assert all(c.visible for c in roles)
    spec = DisplaySpec.from_roles(roles)
    assert spec.x_index is None
    assert spec.y == ((0, "Col 0"), (1, "Col 1"), (2, "Col 2"))


def test_no_columns_is_no_roles():
    assert default_column_roles(0) == []


def test_a_missing_or_blank_name_falls_back_to_a_column_number():
    roles = default_column_roles(3, ("energy", "", ), is_text=True)

    assert [c.name for c in roles] == ["energy", "Col 1", "Col 2"]


# ── with_role activates the role you picked ─────────────────────────── #

def test_picking_a_role_turns_the_column_on():
    """X and Y both take effect on being chosen; only NONE goes dark."""
    off = ColumnRole("signal", Role.NONE, False)

    assert off.with_role(Role.X) == ColumnRole("signal", Role.X, True)
    assert off.with_role(Role.Y) == ColumnRole("signal", Role.Y, True)
    assert ColumnRole("signal", Role.Y, True).with_role(Role.NONE).visible is False


# ── DisplaySpec reads roles, not positions ──────────────────────────── #

def test_display_spec_takes_the_first_x_and_the_visible_ys_in_order():
    roles = [
        ColumnRole("a", Role.Y, False),
        ColumnRole("t", Role.X, True),
        ColumnRole("b", Role.Y, True),
        ColumnRole("c", Role.NONE, False),
        ColumnRole("d", Role.Y, True),
    ]
    spec = DisplaySpec.from_roles(roles)

    assert spec.x_index == 1
    assert spec.x_name == "t"
    assert spec.y == ((2, "b"), (4, "d"))


def test_an_x_column_switched_off_falls_back_to_the_row_index():
    roles = [ColumnRole("t", Role.X, False), ColumnRole("y", Role.Y, True)]

    spec = DisplaySpec.from_roles(roles)

    assert spec.x_index is None
    assert spec.y == ((1, "y"),)
