from combat_math import calculate_damage, apply_damage
from models import Unit, BattleState

def take_turn(state: BattleState, action_a: int, action_b: int = 0) -> BattleState:
    current_agent_hp = state.agent.hp
    current_opp_hp = state.opp.hp

    eff_armor_a = state.agent.armor * 2.0 if action_a == 1 else state.agent.armor
    eff_armor_b = state.opp.armor * 2.0 if action_b == 1 else state.opp.armor

    if action_a == 0: # A attacks
        damage_to_opp = calculate_damage(state.agent.atk, eff_armor_b)
        current_opp_hp = apply_damage(current_opp_hp, damage_to_opp)

    if action_b == 0: #B Simultaneously (both act same tick)
        damage_to_agent = calculate_damage(state.opp.atk, eff_armor_a)
        current_agent_hp = apply_damage(current_agent_hp, damage_to_agent)

    return BattleState(
    agent=Unit(hp=current_agent_hp, atk=state.agent.atk, armor=state.agent.armor),
        opp=Unit(hp=current_opp_hp, atk=state.opp.atk, armor=state.opp.armor)
    )