(define (domain workspace)
	(:requirements :strips :typing :equality :universal-preconditions :negative-preconditions :conditional-effects)
	(:types
		physobj - object
		movable - physobj
		unmovable - physobj
		tool - movable
		box - movable
	)
	(:constants table - unmovable)
	(:predicates
		(inhand ?a - movable)
		(handempty)
		(on ?a - movable ?b - physobj)
		(inworkspace ?a - movable)
		(clear ?a - physobj)
	)
	(:action pick
		:parameters (?a - movable ?b - unmovable)
		:precondition (and
			(on ?a ?b)
			(clear ?a)
			(handempty)
		)
		:effect (and
			(inhand ?a)
			(not (on ?a ?b))
			(not (handempty))
			(not (clear ?a))
		)
	)
	(:action pick_object
		:parameters (?a - movable ?b - movable)
		:precondition (and
			(on ?a ?b)
			(clear ?a)
			(handempty)
		)
		:effect (and
			(inhand ?a)
			(not (on ?a ?b))
			(not (handempty))
			(clear ?b)
			(not (clear ?a))
		)
	)
	(:action place
		:parameters (?a - movable ?b - unmovable)
		:precondition (and
			(inhand ?a)

		)
		:effect (and
			(not (inhand ?a))
			(on ?a ?b)
			(handempty)
			(not (clear ?b))
			(clear ?a)
		)
	)
	(:action place_object
		:parameters (?a - movable ?b - movable)
		:precondition (and
			(inhand ?a)
			(clear ?b)
		)
		:effect (and
			(not (inhand ?a))
			(on ?a ?b)
			(handempty)
			(not (clear ?b))
			(clear ?a)
		)
	)
	(:action pull
		:parameters (?a - movable ?b - movable)
		:precondition (and
			(inhand ?b)
			(on ?a table)
		)
		:effect (and
			(inworkspace ?a)
		)
	)
)
