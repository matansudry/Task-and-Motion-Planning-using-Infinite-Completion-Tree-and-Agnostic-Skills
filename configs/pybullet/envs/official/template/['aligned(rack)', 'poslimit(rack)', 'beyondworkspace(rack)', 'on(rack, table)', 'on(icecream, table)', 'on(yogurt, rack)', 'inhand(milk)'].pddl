(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		milk - box
		yogurt - box
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on icecream table)
		(on yogurt rack)
		(inhand milk)
	)
	(:goal (and))
)
