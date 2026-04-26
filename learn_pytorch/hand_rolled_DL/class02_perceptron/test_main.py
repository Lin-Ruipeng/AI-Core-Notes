import main


# 测试与门逻辑
def test_AND_old():
    assert main.AND_old(0, 0) == 0
    assert main.AND_old(0, 1) == 0
    assert main.AND_old(1, 0) == 0
    assert main.AND_old(1, 1) == 1


def test_AND():
    assert main.AND(0, 0) == 0
    assert main.AND(0, 1) == 0
    assert main.AND(1, 0) == 0
    assert main.AND(1, 1) == 1


def test_NAND():
    assert main.NAND(0, 0) == 1
    assert main.NAND(0, 1) == 1
    assert main.NAND(1, 0) == 1
    assert main.NAND(1, 1) == 0


def test_OR():
    assert main.OR(0, 0) == 0
    assert main.OR(0, 1) == 1
    assert main.OR(1, 0) == 1
    assert main.OR(1, 1) == 1


def test_XOR():
    assert main.XOR(0, 0) == 0
    assert main.XOR(0, 1) == 1
    assert main.XOR(1, 0) == 1
    assert main.XOR(1, 1) == 0
