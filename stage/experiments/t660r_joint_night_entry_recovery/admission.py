"""Pure finite resource admission; no runtime imports or side effects."""
def allowed(root,child,ledger,now,used_bytes):
    checks=dict(
        physics=ledger['physics']+child['max_physics']<=root['night_max_physics'],
        commands=ledger['commands']+child['max_commands']<=root['night_max_commands'],
        main_seconds=ledger['main_seconds']+child['main_seconds']<=root['night_main_seconds'],
        Actor=ledger['Actor_live']+child['actor_cap']<=root['Actor_each_cap'],
        Value=ledger['Value_live']+child['value_cap']<=root['Value_each_cap'],
        storage=used_bytes+child['child_bytes']+root['final_storage_reserve']<=root['max_bytes'],
        native_deadline=now+child['main_seconds']<root['absolute_native_stop'],
        final_deadline=now+child['main_seconds']+child['audit_seconds']+root['finalization_seconds']<root['absolute_all_stop'],
        learning_deadline=(child['kind']!='learn' or now+child['learning_seconds']+child['block_science_reserve_seconds']<root['absolute_learning_stop']))
    return all(checks.values()),checks

def stable_screen(previous_success_ids,screen_success_ids):
    old=set(previous_success_ids);new=set(screen_success_ids)
    return old<=new,sorted(old-new)
