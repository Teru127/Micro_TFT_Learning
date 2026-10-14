from board import make_board, can_place, place, print_board, Unit

def test_oob():
    b = make_board()
    assert can_place(b, 3, 0) is False, "row 3 OOB"
    assert can_place(b, -1, 2) is False, "row -1 OOB"
    assert can_place(b, 0, 4) is False, "col 4 OOB"
    assert can_place(b, 0, 0) is True
    print("test_oob passed")

def test_occupied():
    b = make_board()
    b1 = place(b, Unit("A"), 0, 2)
    assert b1 is not None
    assert b1[0][2].id == "A"
    # second place same cell must fail and not mutate
    b2 = place(b1, Unit("O"), 0, 2)
    assert b2 is None, "occupied should fail"
    assert b1[0][2].id == "A", "board unchanged"
    # original b unchanged (pure)
    assert b[0][2] is None
    print("test_occupied passed")

def test_hardcoded_boards():
    # board 1 : corner
    b1 = make_board()
    b1 = place(b1, Unit("A"), 0, 0)
    print("\nBoard corner (0,0):")
    print_board(b1)
    assert b1[0][0].id == "A"

    # board 2 : center
    b2 = make_board()
    b2 = place(b2, Unit("O"), 1, 1)
    print("\nBoard center (1,1):")
    print_board(b2)
    assert b2[1][1].id == "O"

    # board 3 : 3 units
    b3 = make_board()
    b3 = place(b3, Unit("A"), 0, 2)
    b3 = place(b3, Unit("O"), 1, 1)
    b3 = place(b3, Unit("A"), 2, 3)
    print("\nBoard 3 units:")
    print_board(b3)
    # expected visual:
    # . . A .
    # . O . .
    # . . . A
    assert b3[0][2].id == "A"
    assert b3[1][1].id == "O"
    assert b3[2][3].id == "A"
    print("test_hardcoded_boards passed")


if __name__ == "__main__":
    test_oob()
    test_occupied()
    test_hardcoded_boards()
    print("\nAll tests passed")